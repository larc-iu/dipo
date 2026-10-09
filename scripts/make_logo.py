"""Draw the Dipo logos, docs/logo.svg and docs/logo-square.svg.

A sawtooth depot with one discourse formalism in each bay: RST, PDTB, SDRT,
and dependency. The square version adds a station nameplate reading DIPO under
the building, for slots that want a square picture. Edit the constants below
and rerun to regenerate both SVGs.

Usage: python scripts/make_logo.py [--output docs/logo.svg] [--square-output docs/logo-square.svg]
"""

import argparse
import math
import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, REPO)

from dipo.common.log import wrote  # noqa: E402

# Building colors (IU crimson and cream)
INK = "#2A1E1F"
FRONT = "#F3EDE4"
SIDE = "#BDAE9C"
ROOF = "#9E1B22"
INTERIOR = "#2A1E1F"
JAMB = "#4A3536"
GLASS = "#D5E3EA"
CLOCK = "#FFFFFF"
# Glyph colors: structure lines, discourse units, relations, PDTB Arg1
LINE = "#F3EDE4"
UNIT = "#E9B949"
REL = "#F0705E"
ARG1 = "#8EC5D0"

# SVG units, y grows downward. Each bay is one sawtooth tooth: a vertical edge on
# its left rising to PEAK, then a slope falling to LOW at its right.
GROUND = 290
BAY_W, PEAK, LOW = 117, 92, 186
BAYS = [(40 + i * BAY_W, 40 + (i + 1) * BAY_W, PEAK, LOW) for i in range(4)]
# Screen offset of the building's back plane. This is a parallel projection, not
# true perspective, so every bay, skylight, and door jamb comes out identical.
DEPTH = (54.0, -15.0)
U_WALL = 0.18  # wall thickness at the doorways, as a fraction of DEPTH
CLOCK_BAY = 2
VENT = (1, 0.07, 0.86)  # bay, position down the slope, position back along the depth
VIEWBOX = (8, 54, 590, 250)  # an even margin of about 12 units around the building

# Square version: the building over a nameplate, on a SQUARE x SQUARE canvas.
SQUARE = 600
SQUARE_BUILDING_W = 540
NAMEPLATE_GAP = 30  # between the ground line and the top of the board
NAMEPLATE_PAD = (34, 26)  # around the lettering, horizontal and vertical
NAMEPLATE_SIZE = 130  # lettering size in px (the font's em)
NAMEPLATE_TRACK = 80  # extra space between letters, in font units
# DIPO in Zilla Slab Bold (SIL Open Font License), embedded as outlines so the SVG needs
# no font. Per glyph: advance width, ink bounds (xMin, yMin, xMax, yMax), and path, all
# in font units (1000 per em, y up).
NAMEPLATE_GLYPHS = {
    "D": (
        698,
        (39, 0, 673, 650),
        "M673 328Q673 180 586 90Q499 0 345 0H39V114H109V536H39V650H337Q504 650 588.5 561Q673 472 673 328Z"
        "M529 326Q529 536 334 536H243V114H334Q428 114 478.5 164Q529 214 529 326Z",
    ),
    "I": (352, (39, 0, 313, 650), "M313 0H39V114H109V536H39V650H313V536H243V114H313Z"),
    "P": (
        598,
        (39, 0, 584, 650),
        "M584 438Q584 327 511.5 272.5Q439 218 343 218H243V114H343V0H39V114H109V536H39V650H337Q458 650 521 593"
        "Q584 536 584 438ZM440 436Q440 484 413.5 510Q387 536 320 536H243V331H318Q387 331 413.5 360Q440 389 440 436Z",
    ),
    "O": (
        712,
        (24, -10, 688, 660),
        "M688 326Q688 188 599.5 89Q511 -10 353 -10Q194 -10 109 89Q24 188 24 324Q24 466 114 563Q204 660 357 660"
        "Q513 660 600.5 561.5Q688 463 688 326ZM543 322Q543 426 495 484Q447 542 356 542Q259 542 214 480.5"
        "Q169 419 169 327Q169 238 213 173Q257 108 360 108Q455 108 499 167.5Q543 227 543 322Z",
    ),
}


def proj(p, u=1.0):
    return (p[0] + u * DEPTH[0], p[1] + u * DEPTH[1])


def on_roof(bay, t, u):
    xl, xr, pk, lo = bay
    return proj((xl + t * (xr - xl), pk + t * (lo - pk)), u)


def pts(ps):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in ps)


def poly(ps, fill, sw=4):
    return f'<polygon points="{pts(ps)}" fill="{fill}" stroke="{INK}" stroke-width="{sw}" stroke-linejoin="round"/>'


def arch_path(cx, w, top):
    r = w / 2
    spring = top + r
    return f"M{cx - r:.1f} {GROUND} V{spring:.1f} A{r:.1f} {r:.1f} 0 0 1 {cx + r:.1f} {spring:.1f} V{GROUND} Z"


def front_outline():
    ps = [(BAYS[0][0], GROUND)]
    for xl, xr, pk, lo in BAYS:
        ps += [(xl, pk), (xr, lo)]
    ps.append((BAYS[-1][1], GROUND))
    return ps


def roof_face(xl, xr, pk, lo):
    a, b = (xl, pk), (xr, lo)
    return [a, b, proj(b), proj(a)]


def right_wall():
    xr, lo = BAYS[-1][1], BAYS[-1][3]
    a, b = (xr, GROUND), (xr, lo)
    return [a, b, proj(b), proj(a)]


def skylight(bay, t0=0.2, t1=0.6, u0=0.18, u1=0.78, rows=2, cols=3):
    q = [on_roof(bay, t0, u0), on_roof(bay, t1, u0), on_roof(bay, t1, u1), on_roof(bay, t0, u1)]
    d = ""
    for k in range(1, rows):
        t = t0 + (t1 - t0) * k / rows
        a, b = on_roof(bay, t, u0), on_roof(bay, t, u1)
        d += f"M{a[0]:.1f} {a[1]:.1f} L{b[0]:.1f} {b[1]:.1f} "
    for k in range(1, cols):
        u = u0 + (u1 - u0) * k / cols
        a, b = on_roof(bay, t0, u), on_roof(bay, t1, u)
        d += f"M{a[0]:.1f} {a[1]:.1f} L{b[0]:.1f} {b[1]:.1f} "
    return poly(q, GLASS, sw=3) + f'<path d="{d}" stroke="{INK}" stroke-width="2" stroke-linecap="round"/>'


def vent(bay, t, u, hw=5, rise=8):
    # The pipe's base follows the roof's fall line and has no outline, so the pipe
    # reads as coming out of the roof instead of sitting on it.
    xl, xr, pk, lo = bay
    m = (lo - pk) / (xr - xl)
    x, y = on_roof(bay, t, u)
    bl, br = (x - hw, y - hw * m + 1.5), (x + hw, y + hw * m + 1.5)
    top = bl[1] - rise
    side = f"M{bl[0]:.1f} {bl[1]:.1f} V{top:.1f} H{br[0]:.1f} V{br[1]:.1f}"
    return (
        f'<path d="{side} Z" fill="{SIDE}"/>'
        f'<path d="{side}" fill="none" stroke="{INK}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<path d="M{x - 9:.1f} {top:.1f} A9 7.5 0 0 1 {x + 9:.1f} {top:.1f} Z" fill="{FRONT}" stroke="{INK}" '
        f'stroke-width="3" stroke-linejoin="round"/>'
    )


def gable_spot(bay, door, r):
    # Grid-search the clear wall between the tooth's vertical edge, its slope, and the
    # door arch for the point whose smallest gap to all three is largest, so a circle of
    # radius r there touches nothing.
    xl, xr, pk, lo = bay
    cx, w, top, _ = door
    m = (lo - pk) / (xr - xl)
    best = None
    for i in range(600):
        x = xl + 10 + (i % 30) * 2
        y = pk + 20 + (i // 30) * 3.5
        gaps = (
            x - xl,
            (y - pk - m * (x - xl)) / math.hypot(1, m),
            math.hypot(x - cx, y - (top + w / 2)) - w / 2,
        )
        g = min(gaps) - r
        if best is None or g > best[0]:
            best = (g, x, y)
    return best[1], best[2]


def clock(x, y, r=12):
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{CLOCK}" stroke="{INK}" stroke-width="3.5"/>'
        f'<path d="M{x:.1f} {y - r + 5.5:.1f} V{y:.1f} L{x + 4.5:.1f} {y + 3:.1f}" fill="none" stroke="{INK}" '
        f'stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def markers():
    out = ""
    for name, col in (("mc", REL), ("mm", UNIT), ("mw", LINE)):
        out += (
            f'<marker id="{name}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M1 1.5 L9 5 L1 8.5 Z" fill="{col}" stroke="{col}" stroke-width="1.5" stroke-linejoin="round"/>'
            f"</marker>"
        )
    return out


def line(d, col, marker=None):
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<path d="{d}" fill="none" stroke="{col}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"{m}/>'
    )


def block(cx, y, w, h):
    return f'<rect x="{cx - w / 2:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="3" fill="{UNIT}"/>'


def bar(x, y, w, col, op=1):
    o = f' opacity="{op}"' if op != 1 else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="5" rx="2.5" fill="{col}"{o}/>'


# Glyphs, drawn in front-face coordinates around the doorway center cx.


def g_rst(cx):
    # span bar, a drop to the nucleus, and the satellite's arc into it
    return (
        line(f"M{cx - 30} 226 H{cx + 30}", LINE)
        + line(f"M{cx - 17} 226 V258", LINE)
        + line(f"M{cx + 17} 258 C{cx + 17} 238 {cx - 1} 236 {cx - 9} 244", REL, marker="mc")
        + block(cx - 17, 262, 24, 12)
        + block(cx + 17, 262, 24, 12)
    )


def g_pdtb(cx):
    # running text with Arg1, the connective, and Arg2 highlighted
    x0 = cx - 28
    return (
        bar(x0, 238, 10, LINE, op=0.35)
        + bar(x0 + 13, 238, 30, ARG1)
        + bar(x0 + 46, 238, 10, LINE, op=0.35)
        + bar(x0, 250, 8, LINE, op=0.35)
        + bar(x0 + 11, 250, 9, REL)
        + bar(x0 + 23, 250, 33, UNIT)
        + bar(x0, 262, 22, UNIT)
        + bar(x0 + 25, 262, 31, LINE, op=0.35)
    )


def g_sdrt(cx):
    # a subordinating edge down into a complex discourse unit, coordination inside it
    n1, n2, n3 = (cx - 26, 214), (cx - 16, 262), (cx + 24, 262)
    nodes = "".join(f'<circle cx="{x}" cy="{y}" r="6" fill="{LINE}"/>' for x, y in (n1, n2, n3))
    return (
        f'<rect x="{cx - 32}" y="246" width="70" height="32" rx="8" fill="none" stroke="{LINE}" '
        f'stroke-width="2.5" stroke-dasharray="5 4"/>'
        + line(f"M{n1[0]} {n1[1] + 9} C{n1[0]} 230 {cx - 6} 230 {cx - 6} 240", REL, marker="mc")
        + line(f"M{n2[0] + 9} {n2[1]} H{n3[0] - 11}", UNIT, marker="mm")
        + nodes
    )


def g_dep(cx):
    # a root arrow and head-to-dependent arcs
    xs = (cx - 22, cx, cx + 22)
    return (
        line(f"M{xs[1]} 222 V252", LINE, marker="mw")
        + line(f"M{xs[1] - 3} 258 C{xs[1] - 3} 242 {xs[0]} 242 {xs[0]} 255", REL, marker="mc")
        + line(f"M{xs[1] + 3} 258 C{xs[1] + 3} 242 {xs[2]} 242 {xs[2]} 255", REL, marker="mc")
        + "".join(block(x, 262, 16, 12) for x in xs)
    )


# Doorway per bay: center, width, arch top, glyph. Doors sit left of the bay's center
# so the arch clears the falling roof line.
DOORS = [(b[0] + 52, 84, 182, g) for b, g in zip(BAYS, (g_rst, g_pdtb, g_sdrt, g_dep), strict=True)]


def building():
    body = [f"<defs>{markers()}"]
    for i, (cx, w, top, _) in enumerate(DOORS):
        body.append(f'<clipPath id="d{i}"><path d="{arch_path(cx, w, top)}"/></clipPath>')
    body.append("</defs>")
    rw = right_wall()
    shadow = [rw[0], rw[3], (rw[3][0] + 22, rw[3][1]), (rw[0][0] + 28, GROUND)]
    body.append(f'<polygon points="{pts(shadow)}" fill="{INK}" opacity="0.12"/>')
    # Painter's order: each roof gets its details before the next roof to its right covers it.
    for i, b in enumerate(BAYS):
        body.append(poly(roof_face(*b), ROOF))
        body.append(skylight(b))
        if i == VENT[0]:
            body.append(vent(b, *VENT[1:]))
    body.append(poly(right_wall(), SIDE))
    body.append(poly(front_outline(), FRONT))
    for i, (cx, w, top, glyph) in enumerate(DOORS):
        # The jamb color fills the opening, then the interior is drawn shifted back by the
        # wall thickness, leaving the jamb visible on the left and bottom.
        shift = f'transform="translate({U_WALL * DEPTH[0]:.2f} {U_WALL * DEPTH[1]:.2f})"'
        body.append(f'<path d="{arch_path(cx, w, top)}" fill="{JAMB}"/>')
        body.append(
            f'<g clip-path="url(#d{i})"><path d="{arch_path(cx, w, top)}" fill="{INTERIOR}" {shift}/>'
            f"{glyph(cx + U_WALL * DEPTH[0] / 2)}</g>"
        )
        body.append(
            f'<path d="{arch_path(cx, w, top)}" fill="none" stroke="{INK}" stroke-width="4" stroke-linejoin="round"/>'
        )
    body.append(clock(*gable_spot(BAYS[CLOCK_BAY], DOORS[CLOCK_BAY], 12)))
    body.append(
        f'<line stroke="{INK}" x1="22" y1="{GROUND}" x2="584" y2="{GROUND}" stroke-width="4.5" stroke-linecap="round"/>'
    )
    return "\n".join(body)


def build():
    vx, vy, vw, vh = VIEWBOX
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx} {vy} {vw} {vh}" width="{vw}" height="{vh}">\n'
        + building()
        + "\n</svg>\n"
    )


def nameplate_lettering(text="DIPO"):
    # Lay the glyphs out in font units. Returns the paths and the ink box of the whole word.
    paths, x = [], 0
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for ch in text:
        adv, (gx0, gy0, gx1, gy1), d = NAMEPLATE_GLYPHS[ch]
        paths.append(f'<path transform="translate({x} 0)" d="{d}"/>')
        xmin, ymin = min(xmin, x + gx0), min(ymin, gy0)
        xmax, ymax = max(xmax, x + gx1), max(ymax, gy1)
        x += adv + NAMEPLATE_TRACK
    return "".join(paths), (xmin, ymin, xmax, ymax)


def build_square():
    vx, vy, vw, vh = VIEWBOX
    bw = SQUARE_BUILDING_W
    bh = bw * vh / vw
    k = NAMEPLATE_SIZE / 1000
    lettering, (xmin, ymin, xmax, ymax) = nameplate_lettering()
    tw, th = k * (xmax - xmin), k * (ymax - ymin)
    px, py = NAMEPLATE_PAD
    board_w, board_h = tw + 2 * px, th + 2 * py
    top = (SQUARE - (bh + NAMEPLATE_GAP + board_h)) / 2
    board_x, board_y = (SQUARE - board_w) / 2, top + bh + NAMEPLATE_GAP
    # The board's outline matches the building's 4-unit lines after the building is scaled down.
    board_sw = 4 * bw / vw
    tx = (SQUARE - tw) / 2 - k * xmin
    ty = board_y + py + k * ymax
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SQUARE} {SQUARE}" width="{SQUARE}" height="{SQUARE}">\n'
        f'<svg x="{(SQUARE - bw) / 2:.1f}" y="{top:.1f}" width="{bw}" height="{bh:.1f}" viewBox="{vx} {vy} {vw} {vh}">\n'
        + building()
        + "\n</svg>\n"
        f'<rect x="{board_x:.1f}" y="{board_y:.1f}" width="{board_w:.1f}" height="{board_h:.1f}" rx="12" fill="{ROOF}" '
        f'stroke="{INK}" stroke-width="{board_sw:.2f}"/>\n'
        f'<g fill="{FRONT}" transform="translate({tx:.2f} {ty:.2f}) scale({k} {-k})">{lettering}</g>\n'
        "</svg>\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default=os.path.join(REPO, "docs", "logo.svg"), help="where to write the wide SVG")
    ap.add_argument(
        "--square-output",
        default=os.path.join(REPO, "docs", "logo-square.svg"),
        help="where to write the square SVG, with the nameplate",
    )
    args = ap.parse_args()
    for path, svg in ((args.output, build()), (args.square_output, build_square())):
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        wrote(os.path.abspath(path))


if __name__ == "__main__":
    main()
