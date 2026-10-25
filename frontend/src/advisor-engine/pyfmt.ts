/**
 * Python's number rounding and formatting, reproduced exactly.
 *
 * The engine must give the same scores and sentences as the Python advisor.
 * Python's `round()` and `f"{x:.Nf}"` round the exact binary value half to
 * even (`round(62.5) == 62`, `round(0.15, 1) == 0.1`), while JavaScript's
 * `Math.round` and `toFixed` round ties up. These helpers work on the exact
 * decimal expansion of the double, which `toFixed` returns digit for digit.
 */

/** Extra digits read past the rounding position to tell ties from near-ties. */
const GUARD_DIGITS = 30;

/** Add one to a string of decimal digits ("199" -> "200"). */
function increment(digits: string): string {
  const out = digits.split("");
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i] !== "9") {
      out[i] = String(Number(out[i]) + 1);
      return out.join("");
    }
    out[i] = "0";
  }
  return "1" + out.join("");
}

/**
 * Round a non-negative exact decimal string to `nd` fractional digits, half to even.
 * Returns the digits with a decimal point when `nd > 0`.
 */
function roundDecimalString(exact: string, nd: number): string {
  const [intPart, frac = ""] = exact.split(".");
  const padded = frac.padEnd(nd + 1, "0");
  let digits = intPart + padded.slice(0, nd);
  const rest = padded.slice(nd);
  const first = rest[0];
  const tieOrMore = first === "5" ? (/[1-9]/.test(rest.slice(1)) ? "more" : "tie") : first > "5" ? "more" : "less";
  if (tieOrMore === "more" || (tieOrMore === "tie" && Number(digits[digits.length - 1]) % 2 === 1)) {
    digits = increment(digits);
  }
  if (nd === 0) return digits.replace(/^0+(?=\d)/, "");
  const intDigits = digits.slice(0, digits.length - nd).replace(/^0+(?=\d)/, "") || "0";
  return `${intDigits}.${digits.slice(digits.length - nd)}`;
}

/** The exact decimal expansion of `|x|`, to enough digits to round correctly. */
function exactAbs(x: number, nd: number): string {
  return Math.abs(x).toFixed(Math.min(100, nd + GUARD_DIGITS));
}

/** Python's `round(x, nd)` (and `round(x)` with `nd = 0`). */
export function pyRound(x: number, nd = 0): number {
  if (!Number.isFinite(x)) return x;
  const value = Number(roundDecimalString(exactAbs(x, nd), nd));
  return x < 0 && value !== 0 ? -value : value;
}

/** Python's `f"{x:.{nd}f}"`. */
export function pyFixed(x: number, nd: number): string {
  const body = roundDecimalString(exactAbs(x, nd), nd);
  return x < 0 && Number(body) !== 0 ? `-${body}` : body;
}

/**
 * Python's `f"{x:.0%}"`. CPython multiplies the double by 100 (a float
 * multiplication, not an exact one), then formats it like `.0f`.
 */
export function pyPercent0(x: number): string {
  return `${pyFixed(x * 100, 0)}%`;
}
