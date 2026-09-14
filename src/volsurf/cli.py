"""`volsurf` command line: `report` (README synthetic tables, or the private real-chain row) and `fit` (one surface).

  volsurf report [--readme README.md]
      regenerates the marked blocks of the README from fixed seeds (byte-identical on a second run) and prints the
      measured maxima and the fitter timing to stdout (never into the file)
  volsurf report --data PATH --source {cboe,philippdubach,mztrading} --symbol SYM [--day YYYY-MM-DD] [--rate R]
                 [--quote-date YYYY-MM-DD] [--out FILE] [--header]
      fits the private chain and prints one row in the README's real-chains column format (appended to --out)
  volsurf fit FILE --source ... --symbol ... [--day ...] [--rate R] [--quote-date ...] [--mode auto|parity_line|fixed_discount]
      prints the chain's coverage line and ledger, the term structure per expiry, the skipped slices with reasons,
      the arbitrage notes, and the report dict

`--rate` is the continuously compounded rate per year used to pin the discount on American chains (required there);
`--day` selects the day inside a philippdubach parquet; `--quote-date` overrides the readers' date rules.
Exit status 0 on success, 2 on a usage or data error (message on stderr).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from . import io, report
from .quotes import Chain
from .surface import Surface, fit_surface

__all__ = ["main", "load_chain", "print_surface"]

SOURCES = ("cboe", "philippdubach", "mztrading")


def load_chain(path, source: str, symbol: str | None, day=None, quote_date=None) -> Chain:
    if source == "cboe":
        return io.read_cboe_json(path, quote_date=quote_date)
    if source == "philippdubach":
        if symbol is None or day is None:
            raise ValueError("--symbol and --day are required for philippdubach")
        return io.read_philippdubach(path, symbol, day)
    if source == "mztrading":
        if symbol is None:
            raise ValueError("--symbol is required for mztrading")
        return io.read_mztrading(path, symbol, quote_date=quote_date)
    raise ValueError(f"source must be one of {SOURCES}")


def _parse_date(s: str | None) -> date | None:
    return None if s is None else date.fromisoformat(s)


def print_surface(surface: Surface, chain: Chain, out=None) -> None:
    """Coverage line, reader ledger, term structure, skips, calendar, wide-grid arbitrage notes, report dict.
    `out` defaults to sys.stdout resolved at call time (so a captured stdout sees it)."""
    out = sys.stdout if out is None else out
    cov = chain.coverage()
    print(f"{cov['symbol']} {cov['quote_date']}: {cov['quotes']} quotes, {cov['expiries']} (root, expiry) keys, roots "
          f"{cov['roots']}, two-sided {cov['two_sided']}, zero-bid {cov['zero_bid']}, exercise {cov['exercise']}, "
          f"source {cov['source']}", file=out)
    print("reader ledger: " + str(chain.ledger), file=out)
    ts = surface.term_structure()
    if len(ts):
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
            print(ts.drop(columns=["asymptote_left", "asymptote_right"]).to_string(index=False), file=out)
    for label, reason in surface.skipped:
        print(f"skipped {label}: {reason}", file=out)
    print(f"calendar (quoted k-range): {surface.calendar}", file=out)
    for note in surface.arbitrage().notes:
        print(f"arbitrage (wide grid): {note}", file=out)
    for key, val in surface.report().items():
        if key != "skipped":
            print(f"{key}: {val}", file=out)


def _cmd_report(args) -> int:
    if args.data is None:
        names = report.write_readme_tables(args.readme)
        print(f"README blocks regenerated in {args.readme}: {', '.join(names)}")
        print("measured maxima behind the yes/no cells:")
        for line in report.measured_lines():
            print("  " + line)
        print(report.timing_table(), end="")
        return 0
    chain = load_chain(args.data, args.source, args.symbol, args.day, _parse_date(args.quote_date))
    surface = fit_surface(chain, rate=args.rate, mode=args.mode)
    row = report.private_row(surface)
    text = (report.private_header() if args.header else "") + row
    print(text, end="")
    print(f"machine: {report.machine()}; skipped {len(surface.skipped)}:")
    for label, reason in surface.skipped:
        print(f"  {label}: {reason.splitlines()[0][:160]}")
    if args.out:
        with Path(args.out).open("a") as fh:
            fh.write(text)
    return 0


def _cmd_fit(args) -> int:
    chain = load_chain(args.file, args.source, args.symbol, args.day, _parse_date(args.quote_date))
    surface = fit_surface(chain, rate=args.rate, mode=args.mode)
    print_surface(surface, chain)
    return 0


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--symbol", help="symbol as the source names it (mztrading: _SPX, SPY, AAPL)")
    p.add_argument("--day", help="YYYY-MM-DD, the day inside a philippdubach parquet")
    p.add_argument("--quote-date", help="YYYY-MM-DD, overrides the reader's quote-date rule")
    p.add_argument("--rate", type=float, default=None, help="continuous rate per year; pins the discount on American chains")
    p.add_argument("--mode", choices=("auto", "parity_line", "fixed_discount"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="volsurf", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("report", help="README synthetic tables (no args) or a private real-chain row (--data)")
    r.add_argument("--readme", default="README.md")
    r.add_argument("--data", default=None, help="path of a private chain file")
    r.add_argument("--source", choices=SOURCES, default=None)
    r.add_argument("--symbol", default=None)
    r.add_argument("--day", default=None)
    r.add_argument("--quote-date", default=None)
    r.add_argument("--rate", type=float, default=None)
    r.add_argument("--mode", choices=("auto", "parity_line", "fixed_discount"), default="auto")
    r.add_argument("--out", default=None, help="append the row to this file (keep it gitignored)")
    r.add_argument("--header", action="store_true", help="print the column header before the row")
    r.set_defaults(fn=_cmd_report)
    f = sub.add_parser("fit", help="fit one surface and print its term structure and arbitrage report")
    f.add_argument("file")
    _add_source_args(f)
    f.set_defaults(fn=_cmd_fit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report" and args.data is not None and args.source is None:
        print("volsurf report --data needs --source", file=sys.stderr)
        return 2
    try:
        return int(args.fn(args))
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"volsurf: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
