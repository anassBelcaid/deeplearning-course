from pathlib import Path

OUT = Path(__file__).parent

steps = [
    ("square at d", "1 × (2d = 12)  →  d𝓛/dd = 12", "bottom_left"),
    ("subtract 2", "12 × 1  →  d𝓛/dc = 12", "bottom_right"),
    ("cube", "12 × (3b² = 12)  →  d𝓛/db = 144", "top_right"),
    ("add 1", "144 × 1  →  d𝓛/da = 144", "top_middle"),
    ("square at x", "144 × (2x = 2)  →  d𝓛/dx = 288", "top_left"),
]

nodes = """
<circle class="value" cx="100" cy="135" r="55"/><text class="id" x="100" y="143">x</text>
<rect class="op top_left" x="250" y="80" width="110" height="110" rx="20"/><text class="sym" x="305" y="140">(·)²</text>
<circle class="value" cx="510" cy="135" r="55"/><text class="id" x="510" y="143">a</text>
<rect class="op top_middle" x="660" y="80" width="110" height="110" rx="20"/><text class="sym" x="715" y="140">+1</text>
<circle class="value" cx="920" cy="135" r="55"/><text class="id" x="920" y="143">b</text>
<rect class="op top_right" x="1070" y="80" width="110" height="110" rx="20"/><text class="sym" x="1125" y="140">(·)³</text>
<circle class="value" cx="1125" cy="350" r="55"/><text class="id" x="1125" y="358">c</text>
<rect class="op bottom_right" x="865" y="295" width="110" height="110" rx="20"/><text class="sym" x="920" y="355">−2</text>
<circle class="value" cx="715" cy="350" r="55"/><text class="id" x="715" y="358">d</text>
<rect class="op bottom_left" x="455" y="295" width="110" height="110" rx="20"/><text class="sym" x="510" y="355">(·)²</text>
<circle class="loss" cx="310" cy="350" r="55"/><text class="white" x="310" y="358">𝓛</text>
"""

forward_paths = {
    "top_left": "M155 135H250 M360 135H455",
    "top_middle": "M565 135H660 M770 135H865",
    "top_right": "M975 135H1070 M1125 190V295",
    "bottom_right": "M1070 350H975 M865 350H770",
    "bottom_left": "M660 350H565 M455 350H365",
}

backward_paths = {
    "top_left": "M455 160H360 M250 160H155",
    "top_middle": "M865 160H770 M660 160H565",
    "top_right": "M1125 295V190 M1070 160H975",
    "bottom_right": "M770 375H865 M975 375H1070",
    "bottom_left": "M365 375H455 M565 375H660",
}


def svg(step_number=None, operation=None, formula=None, active=None):
    active_css = f".op.{active}{{stroke:#ef5b45;stroke-width:6}}" if active else ""
    edges = []
    for key, path in forward_paths.items():
        edges.append(f'<path class="forward" d="{path}"/>')
        if key == active:
            edges.append(f'<path class="backward" d="{backward_paths[key]}"/>')

    if step_number is None:
        panel = """
<rect class="panel" x="305" y="455" width="670" height="92" rx="16"/>
<text class="panel-kicker" x="640" y="490">FORWARD VALUES ARE STORED</text>
<text class="panel-main" x="640" y="525">Begin at the loss with sensitivity 1</text>"""
    else:
        panel = f"""
<rect class="panel" x="230" y="445" width="820" height="110" rx="16"/>
<text class="panel-kicker" x="640" y="480">STEP {step_number} · {operation.upper()}</text>
<text class="panel-main" x="640" y="525">{formula}</text>"""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 580" role="img">
<defs>
<marker id="f" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#9aa1a6"/></marker>
<marker id="b" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#ef5b45"/></marker>
<filter id="s" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#0f2a37" flood-opacity=".12"/></filter>
<style>.forward{{fill:none;stroke:#9aa1a6;stroke-width:3;marker-end:url(#f)}}.backward{{fill:none;stroke:#ef5b45;stroke-width:7;marker-end:url(#b)}}.value{{fill:#eaf3fb;stroke:#2878c8;stroke-width:4}}.op{{fill:#fffdf7;stroke:#17212b;stroke-width:4;filter:url(#s)}}.loss{{fill:#ef5b45;stroke:#bd3e2d;stroke-width:4;filter:url(#s)}}.id{{fill:#17212b;font:italic 700 31px Georgia,serif;text-anchor:middle}}.sym{{fill:#17212b;font:700 34px Georgia,serif;text-anchor:middle;dominant-baseline:middle}}.white{{fill:#fff;font:italic 700 32px Georgia,serif;text-anchor:middle}}.panel{{fill:#fffdf7;stroke:#ef5b45;stroke-width:4;filter:url(#s)}}.panel-kicker{{fill:#657079;font:700 17px 'Courier New',monospace;letter-spacing:1px;text-anchor:middle}}.panel-main{{fill:#17212b;font:700 29px Arial,sans-serif;text-anchor:middle}}{active_css}</style>
</defs>
<rect width="1280" height="580" fill="#f4f0e7"/>
{''.join(edges)}
{nodes}
{panel}
</svg>"""


(OUT / "scalar-backprop-step-0.svg").write_text(svg(), encoding="utf-8")
for index, (operation, formula, active) in enumerate(steps, start=1):
    (OUT / f"scalar-backprop-step-{index}.svg").write_text(
        svg(index, operation, formula, active), encoding="utf-8"
    )
