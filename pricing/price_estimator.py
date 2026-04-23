"""
Price estimation from bag attributes.

Usage:
    python price_estimator.py --model Birkin --size 30 --leather Togo --color Noir
    python price_estimator.py --model Birkin --size 25 --leather Epsom --color Craie --hardware Palladium --condition "Pre-Loved"
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from price_data import estimate_price


def main():
    parser = argparse.ArgumentParser(description="Hermès Price Estimator")
    parser.add_argument("--model",     type=str, required=True, help="Bag model (Birkin)")
    parser.add_argument("--size",      type=str, required=True, help="Size (25, 30, 35)")
    parser.add_argument("--leather",   type=str, required=True, help="Leather (Togo, Epsom, Crocodile, ...)")
    parser.add_argument("--color",     type=str, required=True, help="Color (Noir, Gold, Craie, ...)")
    parser.add_argument("--hardware",  type=str, default=None,  help="Hardware (Gold, Palladium, Rose Gold, ...)")
    parser.add_argument("--condition", type=str, default=None,  help="Condition (Box Fresh, Pre-Loved, Excellent, Good, Fair)")
    args = parser.parse_args()

    print("=" * 52)
    print("  Hermès Price Estimator")
    print("=" * 52)
    print(f"  Model:     {args.model}")
    print(f"  Size:      {args.size}")
    print(f"  Leather:   {args.leather}")
    print(f"  Color:     {args.color}")
    if args.hardware:  print(f"  Hardware:  {args.hardware}")
    if args.condition: print(f"  Condition: {args.condition}")

    result = estimate_price(
        model=args.model,
        size=args.size,
        leather=args.leather,
        color=args.color,
        hardware=args.hardware,
        condition=args.condition,
    )

    if "error" in result:
        print(f"\n  ⚠  {result['error']}")
        return

    print(f"\n  Price Estimate (EUR)")
    print(f"  {'─' * 25}")
    print(f"  Low:   €{result['low']:>8,}")
    print(f"  Mid:   €{result['mid']:>8,}")
    print(f"  High:  €{result['high']:>8,}")
    print(f"\n  Based on {result['comparables']} comparable listings")
    if result.get("multiplier", 1.0) != 1.0:
        print(f"  Total multiplier applied: ×{result['multiplier']}")
    for note in result.get("notes", []):
        print(f"  ℹ  {note}")
    print(f"\n  Source: Love Luxury (April 2026)")


if __name__ == "__main__":
    main()