"""
inference/inventory.py
Simulated planogram + warehouse inventory database.

In a real system this would connect to a retailer's ERP/WMS.
For the prototype we use a simple in-memory dict.
"""

# ── Simulated planogram ───────────────────────────────────────────────────────
# Maps shelf section → expected product
PLANOGRAM = {
    "section_A": "Coca Cola 500ml",
    "section_B": "Pepsi 500ml",
    "section_C": "Water Still 1L",
    "section_D": "Orange Juice 1L",
    "section_E": "Milk 1L",
    "section_F": "Yogurt Natural",
    "section_G": "Butter 250g",
    "section_H": "Cheese Sliced",
}

# ── Simulated warehouse stock (units available) ───────────────────────────────
WAREHOUSE = {
    "Coca Cola 500ml":  24,
    "Pepsi 500ml":       0,   # out of stock → triggers purchase order
    "Water Still 1L":   50,
    "Orange Juice 1L":  12,
    "Milk 1L":           8,
    "Yogurt Natural":    0,   # out of stock → triggers purchase order
    "Butter 250g":      15,
    "Cheese Sliced":     6,
}


def get_product_for_section(section: str) -> str:
    """Returns the product that should be in a given shelf section."""
    return PLANOGRAM.get(section, "Unknown Product")


def check_stock(product: str) -> int:
    """Returns warehouse stock level for a product."""
    return WAREHOUSE.get(product, 0)


def process_empty_gap(section: str) -> dict:
    """
    Given an empty shelf section, decides whether to:
    - Create a restocking task (warehouse has stock)
    - Trigger a purchase order (warehouse is empty)
    """
    product = get_product_for_section(section)
    stock   = check_stock(product)

    if stock > 0:
        action = "RESTOCK_TASK"
        message = f"📦 Task created: Restock '{product}' in {section} ({stock} units available in warehouse)"
    else:
        action = "PURCHASE_ORDER"
        message = f"🛒 Purchase order triggered: '{product}' is out of stock in warehouse — order from supplier"

    return {
        "section":  section,
        "product":  product,
        "stock":    stock,
        "action":   action,
        "message":  message,
    }


def process_all_gaps(empty_sections: list) -> list:
    """Processes a list of empty shelf sections and returns actions."""
    return [process_empty_gap(s) for s in empty_sections]
