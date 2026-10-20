"""
tools.py - The tools (functions) each specialist agent is allowed to call.

Each agent only gets the tools for its own job:
  Order Status Agent      -> get_order_details, update_order_quantity
  Refund Policy Agent     -> get_refund_policy, check_refund_eligibility, create_return_request
  Product Recommendation  -> list_products, get_product_details

The data comes from the starter repo's CSV files in data/ (orders, pricing)
plus a product description table and a refund policy document.
"""

import os
import random
from datetime import date

import pandas as pd
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The sample data is from fall 2024, so the store uses a fixed "today" date.
# This makes order status and refund windows behave the same every run.
STORE_TODAY = date.fromisoformat(os.getenv("STORE_DATE", "2024-10-20"))

# Return windows in days (must match data/refund_policy.md)
STANDARD_RETURN_DAYS = 30
GAMING_RETURN_DAYS = 15

# Load the data into memory once
orders_df = pd.read_csv(os.path.join(DATA_DIR, "Laptop Orders.csv"))
pricing_df = pd.read_csv(os.path.join(DATA_DIR, "Laptop pricing.csv"))
descriptions_df = pd.read_csv(os.path.join(DATA_DIR, "Laptop descriptions.csv"))
products_df = pricing_df.merge(descriptions_df, on="Name")  # one table with price + specs
with open(os.path.join(DATA_DIR, "refund_policy.md"), encoding="utf-8") as f:
    REFUND_POLICY = f.read()

# Return requests created during this session (return id -> details)
return_requests = {}


def _find_order(order_id: str):
    """Return the order row as a dict, or None if the order ID does not exist."""
    order_id = order_id.strip().upper()
    match = orders_df[orders_df["Order ID"] == order_id]
    return None if match.empty else match.iloc[0].to_dict()


def _order_status(delivery_date: str) -> str:
    """Delivered if the delivery date has passed, otherwise in transit."""
    d = date.fromisoformat(delivery_date)
    if d <= STORE_TODAY:
        return f"Delivered on {delivery_date}"
    return f"In transit - arriving {delivery_date} ({(d - STORE_TODAY).days} days from today)"


# ---------------------------------------------------------------- Order tools
@tool
def get_order_details(order_id: str) -> str:
    """Look up a laptop order by its exact order ID (format ORD-1234).
    Returns the laptop ordered, quantity, delivery date and current status.
    Returns 'ORDER_NOT_FOUND' if no order has that ID."""
    order = _find_order(order_id)
    if order is None:
        return "ORDER_NOT_FOUND"
    return (
        f"Order {order['Order ID']}: {order['Quantity Ordered']} x {order['Product Ordered']} | "
        f"Delivery date: {order['Delivery Date']} | Status: {_order_status(order['Delivery Date'])}"
    )


@tool
def update_order_quantity(order_id: str, new_quantity: int) -> str:
    """Change the quantity of laptops in an order. Only allowed for orders that
    have NOT been delivered yet. new_quantity must be between 1 and 10."""
    order = _find_order(order_id)
    if order is None:
        return "ORDER_NOT_FOUND"
    if date.fromisoformat(order["Delivery Date"]) <= STORE_TODAY:
        return "UPDATE_REJECTED: this order was already delivered and can no longer be changed."
    if not 1 <= int(new_quantity) <= 10:
        return "UPDATE_REJECTED: quantity must be between 1 and 10."
    old_qty = order["Quantity Ordered"]
    orders_df.loc[orders_df["Order ID"] == order["Order ID"], "Quantity Ordered"] = int(new_quantity)
    return f"UPDATED: order {order['Order ID']} quantity changed from {old_qty} to {new_quantity}."


# --------------------------------------------------------------- Refund tools
@tool
def get_refund_policy() -> str:
    """Return the full TechNova refund and return policy document."""
    return REFUND_POLICY


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether a specific order can still be returned for a refund,
    based on its delivery date, today's date and the product's return window."""
    order = _find_order(order_id)
    if order is None:
        return "ORDER_NOT_FOUND"
    product = order["Product Ordered"]
    delivered = date.fromisoformat(order["Delivery Date"])
    if delivered > STORE_TODAY:
        return (
            f"Order {order['Order ID']} ({product}) has NOT been delivered yet. "
            "It can be cancelled now for a full refund at no cost."
        )
    window = GAMING_RETURN_DAYS if product.startswith("OmegaPro") else STANDARD_RETURN_DAYS
    days_since = (STORE_TODAY - delivered).days
    if days_since <= window:
        return (
            f"ELIGIBLE: order {order['Order ID']} ({product}) was delivered {days_since} days ago. "
            f"Return window is {window} days, so {window - days_since} days remain."
        )
    return (
        f"NOT ELIGIBLE: order {order['Order ID']} ({product}) was delivered {days_since} days ago, "
        f"which is past the {window}-day return window."
    )


@tool
def create_return_request(order_id: str, reason: str) -> str:
    """Open a return/refund request (RMA) for an ELIGIBLE order. Only call this
    after the customer confirms they want to return it. Returns the return ID."""
    order = _find_order(order_id)
    if order is None:
        return "ORDER_NOT_FOUND"
    eligibility = check_refund_eligibility.invoke({"order_id": order_id})
    if eligibility.startswith("NOT ELIGIBLE"):
        return "RETURN_REJECTED: " + eligibility
    rma_id = f"RMA-{random.randint(10000, 99999)}"
    return_requests[rma_id] = {"order_id": order["Order ID"], "reason": reason}
    return (
        f"RETURN_CREATED: {rma_id} for order {order['Order ID']} (reason: {reason}). "
        "A prepaid return label will be emailed within 24 hours."
    )


# -------------------------------------------------------------- Product tools
@tool
def list_products(max_price: int = 0, max_shipping_days: int = 0) -> str:
    """List laptops in the catalog with price, shipping time, category and what
    each is best for. Optional filters: max_price in USD and max_shipping_days.
    Use 0 to mean 'no limit'."""
    df = products_df
    if max_price:
        df = df[df["Price"] <= max_price]
    if max_shipping_days:
        df = df[df["ShippingDays"] <= max_shipping_days]
    if df.empty:
        return "NO_MATCHING_PRODUCTS"
    lines = [
        f"- {r['Name']}: ${r['Price']}, ships in {r['ShippingDays']} days, "
        f"{r['Category']}, best for {r['Best For']}"
        for _, r in df.iterrows()
    ]
    return "\n".join(lines)


@tool
def get_product_details(laptop_name: str) -> str:
    """Get full specs, price and shipping time for one laptop. Partial names
    work, e.g. 'Spectra' or 'omegapro'."""
    match = products_df[products_df["Name"].str.contains(laptop_name.strip(), case=False, regex=False)]
    if match.empty:
        return "PRODUCT_NOT_FOUND - we only sell: " + ", ".join(products_df["Name"])
    r = match.iloc[0]
    return (
        f"{r['Name']} ({r['Category']}) - ${r['Price']}, ships in {r['ShippingDays']} days.\n"
        f"Best for: {r['Best For']}\nSpecs: {r['Description']}"
    )


ORDER_TOOLS = [get_order_details, update_order_quantity]
REFUND_TOOLS = [get_refund_policy, check_refund_eligibility, create_return_request]
PRODUCT_TOOLS = [list_products, get_product_details]
