#!/usr/bin/env python3
"""Approximate sample sizes for comparing two binomial rates."""

import argparse
import math


Z_BY_CONFIDENCE = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}

Z_BY_POWER = {
    0.70: 0.5244005127080409,
    0.80: 0.8416212335729143,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
}


def nearest_z(value, table, name):
    if value in table:
        return table[value]
    known = ", ".join(str(k) for k in sorted(table))
    raise SystemExit(f"Unsupported {name}={value}. Supported values: {known}")


def two_proportion_test_n(p1, p2, alpha_z, power_z):
    """Normal-approx n per group for a two-sided two-proportion test."""
    delta = abs(p1 - p2)
    if delta <= 0:
        return math.inf
    pbar = (p1 + p2) / 2
    term_alpha = alpha_z * math.sqrt(2 * pbar * (1 - pbar))
    term_power = power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return ((term_alpha + term_power) / delta) ** 2


def nonoverlap_ci_n(p1, p2, z):
    """Rough n per group for non-overlapping Wald-style confidence intervals."""
    delta = abs(p1 - p2)
    if delta <= 0:
        return math.inf
    spread = z * (math.sqrt(p1 * (1 - p1)) + math.sqrt(p2 * (1 - p2)))
    return (spread / delta) ** 2


def fmt_n(value):
    if math.isinf(value):
        return "infinite"
    return str(math.ceil(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=float, required=True,
                        help="First observed rate, e.g. 0.017.")
    parser.add_argument("--p2", type=float, required=True,
                        help="Second observed rate, e.g. 0.008.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for two-sided test/CI. Supported: 0.8, 0.9, 0.95, 0.98, 0.99.")
    parser.add_argument("--power", type=float, default=0.80,
                        help="Power for two-proportion test. Supported: 0.7, 0.8, 0.9, 0.95.")
    parser.add_argument("--label1", default="rate_1")
    parser.add_argument("--label2", default="rate_2")
    args = parser.parse_args()

    if not (0 <= args.p1 <= 1 and 0 <= args.p2 <= 1):
        raise SystemExit("p1 and p2 must be probabilities in [0, 1]")
    alpha_z = nearest_z(args.confidence, Z_BY_CONFIDENCE, "confidence")
    power_z = nearest_z(args.power, Z_BY_POWER, "power")
    test_n = two_proportion_test_n(args.p1, args.p2, alpha_z, power_z)
    ci_n = nonoverlap_ci_n(args.p1, args.p2, alpha_z)

    print(f"{args.label1}: {args.p1:.4f}")
    print(f"{args.label2}: {args.p2:.4f}")
    print(f"absolute difference: {abs(args.p1 - args.p2):.4f}")
    print()
    print(
        f"Approx n per model for {args.confidence:.0%} two-sided two-proportion test "
        f"with {args.power:.0%} power: {fmt_n(test_n)}"
    )
    print(
        f"Rough n per model for non-overlapping {args.confidence:.0%} CIs: {fmt_n(ci_n)}"
    )
    print()
    print("Treat these as planning estimates; judge noise and prompt clustering can require more.")


if __name__ == "__main__":
    main()
