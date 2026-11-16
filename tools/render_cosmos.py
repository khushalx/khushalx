#!/usr/bin/env python3
"""Build both Crimson Cosmos covers and the divider from local inputs."""

import base64
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'


def smooth(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def noise(x, y):
    """Continuous, deterministic value noise in art or disk coordinates."""
    i, j = np.floor(x), np.floor(y)
    fx, fy = smooth(0, 1, x - i), smooth(0, 1, y - j)
    i, j = i.astype(np.int32).view(np.uint32), j.astype(np.int32).view(np.uint32)

    def h(dx, dy):
        key = ((i + np.uint32(dx)) * np.uint32(374761393)
               + (j + np.uint32(dy)) * np.uint32(668265263))
        key = (key ^ (key >> 13)) * np.uint32(1274126177)
        return ((key ^ (key >> 16)) & 65535).astype(np.float32) / 65535

    return ((h(0, 0) * (1 - fx) + h(1, 0) * fx) * (1 - fy)
            + (h(0, 1) * (1 - fx) + h(1, 1) * fx) * fy)


def turbulence(x, y, octaves=5):
    out, amplitude = 0, .53
    for _ in range(octaves):
        out += amplitude * noise(x, y)
        x, y = 1.8 * x + .91 * y + 11.4, -.91 * x + 1.8 * y - 8.3
        amplitude *= .48
    return out


def geodesics():
    """Tabulate inverse radius u(phi,b); Schwarzschild units with G=c=M=1.

    Integrate u'' = 3u² - u from a distant camera with u(0)=0 and
    u'(0)=1/b. A ray stops contributing at the horizon or on escape.
    """
    db, dp = .004, .0035
    b = np.arange(.004, 44 + db, db, dtype=np.float64)
    table = np.empty((int(3 * np.pi / dp) + 2, b.size), dtype=np.float32)
    u, v = np.zeros_like(b), 1 / b
    alive = np.ones(b.size, dtype=bool)
    table[0] = u

    def accel(r):
        return r * (3 * r - 1)

    for i in range(1, len(table)):
        k1 = accel(u)
        k2 = accel(u + .5 * dp * v)
        k3 = accel(u + .5 * dp * (v + .5 * dp * k1))
        k4 = accel(u + dp * (v + .5 * dp * k2))
        u += dp * (v + dp / 6 * (k1 + k2 + k3))
        v += dp / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        alive &= (u > 0) & (u < .5)
        table[i] = np.where(alive, u, 0)
        u[~alive], v[~alive] = 0, 0
    return table, dp, db


def disk_light(x, y):
    table, dp, db = geodesics()
    b = np.maximum(np.hypot(x, y), .01)
    nx, ny = x / b, y / b
    # The disk is seen 9° above its equator. Consecutive plane crossings
    # show the direct disk and its gravitationally lensed higher images.
    elevation = np.deg2rad(9)
    se, ce = np.sin(elevation), np.cos(elevation)
    phase = np.mod(np.arctan2(-se, ny * ce), np.pi)
    ib = np.clip(b / db - 1, 0, table.shape[1] - 2.001)
    left = ib.astype(np.int32)
    mix = ib - left
    light, heat = np.zeros_like(x), np.zeros_like(x)
    transmit = np.ones_like(x)

    for order in range(3):
        a = phase + order * np.pi
        ip = a / dp
        lower = ip.astype(np.int32)
        frac = ip - lower
        u1 = table[lower, left] * (1 - mix) + table[lower, left + 1] * mix
        u2 = table[lower + 1, left] * (1 - mix) + table[lower + 1, left + 1] * mix
        r = 1 / np.maximum(u1 * (1 - frac) + u2 * frac, .01)
        sa, ca = np.sin(a), np.cos(a)
        px, py = r * nx * sa, r * (ce * ca - se * ny * sa)
        theta = np.arctan2(py, px)
        # Wind the texture into turbulent shear layers, with no clean
        # concentric outlines. The approaching side is Doppler boosted.
        wind = 2.6 * np.log(r / 6)
        cx, cy = np.cos(theta + wind), np.sin(theta + wind)
        broad = turbulence(cx * 4.5 + r * .37, cy * 4.5 - r * .13, 4)
        fine = turbulence(cx * 11.8 + r * 2.9, cy * 11.8 - r * 1.7, 4)
        wave = .5 + .5 * np.sin(r * 12.4 + broad * 10 + fine * 2.8 + 2 * np.cos(theta + r * .27))
        shreds = .5 + .5 * np.sin(r * 26.5 + broad * 15 + fine * 4)
        filaments = .24 + .76 * (wave ** 2 * .61 + shreds * .12 + fine * .59)
        hot = (np.clip(broad * .77 + fine * .42, 0, 1) * 1.18) ** 2.5
        edge = 5.68 + .37 * np.sin(theta * 3 + .3) + .23 * np.cos(theta * 7)
        outer = 22 + 2.4 * np.sin(2 * theta + .8) + 1.3 * np.cos(5 * theta)
        taper = smooth(edge, edge + 1.3, r) * (1 - smooth(14.5, outer, r))
        speed = 1 / np.sqrt(np.maximum(r - 2, 1))
        doppler = 1 / (1 + .84 * speed * np.cos(theta - .22)) ** 3
        hot_inner = np.exp(-((r - 7.25) / 2.0) ** 2) * (.49 + fine * .75)
        density = taper * np.exp(-(r - 6) * .112) * (filaments * (hot + .25) + hot_inner * .32)
        emission = density * doppler * (1, .92, .64)[order] * transmit
        light += emission
        heat += emission * (hot_inner * .37 + fine * .35 + doppler * .14)
        transmit *= 1 - .91 * taper * smooth(.01, .15, density)

    return b, light, heat


def blur(field, sigma):
    # Separable Gaussian, using an FFT so the dim scattering can stay in float.
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.rfftfreq(field.shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(field) * kernel, s=field.shape, axes=(0, 1)).astype(np.float32)


def artwork():
    """Return desktop/mobile crops of the shared ray-traced exposure."""
    w, h = 1320, 980
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # Camera rotated 13° so the disk cuts the negative space diagonally.
    dx, dy = (xx - 686) / 32, (478 - yy) / 32
    roll = np.deg2rad(13)
    x, y = dx * np.cos(roll) + dy * np.sin(roll), dy * np.cos(roll) - dx * np.sin(roll)
    b, disk, heat = disk_light(x, y)
    ring = np.exp(-((b - 5.208) / .033) ** 2) * .16 * (1 + .73 * np.cos(np.arctan2(y, x) + 2.5))
    ring *= .6 + .5 * noise(x * 3, y * 3)
    disk += ring
    small, mid, far = blur(disk, 2.3), blur(disk, 12), blur(disk, 38)
    core = 1 - np.exp(-1.45 * disk)
    warmth = np.clip((1 - np.exp(-1.25 * heat)) * 1.4, 0, 1)

    # Faint, warped clouds well behind the accretion disk. Mask the event
    # shadow out of the sky before adding optical scatter in front of it.
    warp = turbulence(xx / 150 + 12, yy / 106 - 85, 4)
    sky = turbulence(xx / 211 + warp * 2.7, yy / 94 - warp * 3.4, 6)
    veil = turbulence(x * .32 + warp * 2 + 98, y * .68 - warp * 2, 5)
    plume = np.exp(-((y - .11 * x - 2.0 * np.cos(x * .14)) / (4.3 + .08 * np.abs(x))) ** 2)
    nebula = (sky ** 2.7 * .6 + np.maximum(veil - .20, 0) ** 1.45) * plume
    nebula *= np.exp(-((x + 3) / 24) ** 2) * smooth(5.18, 5.46, b)
    clouds = (warp ** 4) * np.exp(-((dx - 6) / 19) ** 2 - ((dy + 1) / 11) ** 2) * smooth(5.18, 5.46, b)

    red = 5 + core * 207 + small * 56 + mid * 87 + far * 94 + nebula * 72 + clouds * 18
    green = 5 + core * (14 + warmth * 78) + small * 5 + mid * 8 + far * 8 + nebula * 5 + clouds * 6
    blue = 7 + core * (19 + warmth * 43) + small * 5 + mid * 9 + far * 10 + nebula * 9 + clouds * 12
    scatter = np.dstack((red, green, blue))

    rng = np.random.default_rng(79163)
    # Unresolved stars and dust give the exposure fine detail. Keep the
    # interior of the shadow clear and the brightest stars scarce.
    for ix, iy in zip(rng.integers(3, w - 3, 3000), rng.integers(3, h - 3, 3000)):
        if b[iy, ix] < 5.28:
            continue
        strength = rng.random() ** 5
        value = (6 + 147 * strength) * (1 - .55 * core[iy, ix])
        shade = np.array((1, .65, .56) if rng.random() < .53 else (.86, .88, 1)) * value
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                scatter[iy + sy, ix + sx] += shade * np.exp(-(sx * sx + sy * sy) / (.40 + strength))
    grain = rng.normal(0, .74, size=(h, w, 1))
    img = Image.fromarray(np.clip(scatter + grain, 0, 255).astype(np.uint8))
    return img


@lru_cache(maxsize=None)
def font(weight):
    face = TTFont(ROOT / 'tools/fonts/Manrope.ttf')
    return instantiateVariableFont(face, {'wght': weight}, inplace=True)


def lettering(label, x, baseline, size, color, weight=650, tracking=0, align='left'):
    """Outline Manrope glyphs; no browser font load or machine-specific metrics."""
    face = font(weight)
    glyphs, cmap = face.getGlyphSet(), face.getBestCmap()
    scale = size / face['head'].unitsPerEm
    names = [cmap[ord(c)] for c in label]
    advances = [glyphs[name].width * scale + tracking for name in names]
    if align == 'right':
        x -= sum(advances) - tracking
    parts = []
    for name, advance in zip(names, advances):
        pen = SVGPathPen(glyphs, ntos=lambda n: f'{n:.2f}'.rstrip('0').rstrip('.'))
        glyphs[name].draw(pen)
        if pen.getCommands():
            parts.append(f'<path transform="translate({x:.2f} {baseline}) scale({scale:.6f} {-scale:.6f})" d="{pen.getCommands()}"/>')
        x += advance
    return f'<g fill="{color}" aria-label="{label.replace("&", "&amp;")}">{"".join(parts)}</g>'


def picture_payload(img, layout):
    """Crop to the visible frame, then embed one optimized photographic layer."""
    ix, iy, scale = layout['image']
    w, h = layout['size']
    left = max(0, int(-ix / scale))
    top = max(0, int(-iy / scale))
    right = min(img.width, int(np.ceil((w - ix) / scale)))
    bottom = min(img.height, int(np.ceil((h - iy) / scale)))
    buffer = io.BytesIO()
    img.crop((left, top, right, bottom)).save(buffer, 'JPEG', quality=83, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return (f'<image x="{ix + left * scale:.2f}" y="{iy + top * scale:.2f}" '
            f'width="{(right-left)*scale:.2f}" height="{(bottom-top)*scale:.2f}" '
            f'href="data:image/jpeg;base64,{encoded}"/>')


MOTION = '''<style>
@media (prefers-reduced-motion: no-preference) {
  .drift { animation: drift 32s ease-in-out infinite; }
  .beacon { animation: breathe 7s ease-in-out infinite; }
  .twinkle { animation: breathe 9s ease-in-out -4s infinite; }
  .tracer { animation: trace 30s linear infinite; }
  .tracer-b { animation: trace 43s linear -17s infinite reverse; }
  .orbit { animation: precess 48s ease-in-out infinite; }
}
@keyframes breathe { 0%,100% {opacity:.42} 50% {opacity:.87} }
@keyframes drift { 0%,100% {transform:translate(0,0)} 50% {transform:translate(10px,-4px)} }
@keyframes trace { to {stroke-dashoffset:-100} }
@keyframes precess { 0%,100% {transform:rotate(0)} 50% {transform:rotate(1.3deg)} }
:root:target .drift, :root:target .beacon, :root:target .twinkle,
:root:target .tracer, :root:target .tracer-b, :root:target .orbit { animation:none; }
</style>'''


def sky_marks(layout):
    w, h = layout['size']
    ix, iy, scale = layout['image']
    cx, cy = ix + 686 * scale, iy + 478 * scale
    rad = 32 * scale
    rng = np.random.default_rng(12229)
    dust = []
    for _ in range(88):
        x = rng.uniform(-rad * 15, rad * 15)
        y = rng.normal(0, rad * 1.24)
        if ((x / rad) ** 2 + (y / rad) ** 2) < 30 or cx + x < layout['stars_start']:
            continue
        dust.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rng.uniform(.35, 1.15):.2f}" '
                    f'fill="{rng.choice(["#F5F1EF", "#FF7761", "#DC143C"])}" opacity="{rng.uniform(.18, .53):.2f}"/>')
    paths = []
    for sx, sy, tilt in ((19, 5.1, 1), (16.1, 6.7, -.8)):
        paths.append(f'<ellipse rx="{rad * sx:.2f}" ry="{rad * sy:.2f}" transform="rotate({tilt})"/>')
    tracer = f'<ellipse rx="{rad * 10.5:.2f}" ry="{rad * 1.85:.2f}" pathLength="100"'
    stars = []
    for index, (x, y, bright) in enumerate(layout['stars']):
        arms = 6 if bright else 3
        stars.append(f'<g class="{"beacon" if index == 0 else "twinkle"}" opacity=".66" transform="translate({x} {y})">'
                     f'<path d="M{-arms} 0H{arms}M0 {-arms}V{arms}" stroke="#F58A73" stroke-width=".75"/>'
                     f'<circle r="{1.55 if bright else .75}" fill="#F5F1EF"/></g>')
    return (f'<g clip-path="url(#art-safe)">'
            f'<g transform="translate({cx:.2f} {cy:.2f}) rotate(-13)">'
            f'<g class="orbit" fill="none" stroke="#D99389" stroke-width=".75" opacity=".24">{"".join(paths)}</g>'
            f'<g class="drift">{"".join(dust)}</g>'
            f'<g fill="none" stroke-linecap="round">{tracer} class="tracer" stroke="#FF9B78" '
            f'stroke-width="1.25" stroke-dasharray=".12 31 .05 68.83" opacity=".48"/>'
            f'{tracer} class="tracer-b" stroke="#FF5039" stroke-width="1.5" '
            f'stroke-dasharray=".15 47 .06 52.79" opacity=".39"/></g>'
            f'</g>{"".join(stars)}</g>')


def typography(layout):
    w, h = layout['size']
    mobile = layout['mobile']
    x = 39 if mobile else 66
    rx = w - x
    guide = 78 if mobile else 79
    bottom = h - (51 if mobile else 52)
    name_size = 110 if mobile else 122
    name_y = (187, 278) if mobile else (233, 337)
    subtitle = ([lettering('COMPUTER SCIENCE &', x, 323, 24, '#F5F1EF', 760, 1.1),
                 lettering('ARTIFICIAL INTELLIGENCE', x, 352, 24, '#F5F1EF', 760, 1.1)] if mobile else
                [lettering('COMPUTER SCIENCE & ARTIFICIAL INTELLIGENCE', x, 391, 17, '#F5F1EF', 760, 1.45)])
    deck_y = 390 if mobile else 425
    footer_y = h - 24
    warm, grey = '#F5F1EF', '#A6A2A3'
    # Manrope outlines, exception: purely decorative plotted ticks below.
    pieces = [
        lettering('CRIMSON COSMOS', x + 25, 52, 15, warm, 760, 3.1),
        lettering('FIELD NOTES  /  01', rx, 52, 13, grey, 660, 1.7, 'right'),
        lettering('KHUSHAL', x - 5, name_y[0], name_size, warm, 790, -4.2),
        lettering('DANGAR', x - 3, name_y[1], name_size, warm, 790, -4.2),
        *subtitle,
        lettering('Exploring complexity. Engineering possibilities.', x, deck_y, 23 if mobile else 20, '#BEB9B9', 530),
        lettering('FIG. 01   /   EVENT HORIZON', x, footer_y, 13 if mobile else 12, grey, 700, 1.4),
        lettering('GRAVITATIONAL LENSING', rx, footer_y, 12 if mobile else 11.5, grey, 650, 1.5, 'right'),
    ]
    ruler = []
    for index in range(11):
        tick = w - x - index * (9 if mobile else 13)
        ruler.append(f'M{tick} {bottom}v{-7 if index % 5 == 0 else -3.8}')
    markers = (f'<path d="M{x} {guide}H{rx}M{x} {bottom}H{rx}{"".join(ruler)}" '
               f'fill="none" stroke="#C4A7A5" stroke-opacity=".37" stroke-width="1"/>'
               f'<path d="M{x} {guide}h45" stroke="#FF303F" stroke-width="1.5"/>'
               f'<path d="M{x} 46l5 -5 5 5-5 5z" fill="none" stroke="#FF4C52" stroke-width="1.2"/>'
               f'<circle cx="{x + 5}" cy="46" r="1.15" fill="#F5F1EF"/>')
    if not mobile:
        markers += ('<path d="M1214 112v76m0-76h-11m11 19h-5m5 19h-5m5 19h-5m5 19h-11" '
                    'fill="none" stroke="#BF9188" stroke-width=".8" opacity=".62"/>'
                    '<path d="M1167 437h20l20-22" fill="none" stroke="#BF9188" stroke-width=".8" opacity=".64"/>')
        pieces.extend([lettering('+15°', 1195, 119, 11, grey, 650, 1.3, 'right'),
                       lettering('−15°', 1195, 189, 11, grey, 650, 1.3, 'right')])
    return markers + ''.join(pieces)


def cover(img, layout):
    w, h = layout['size']
    mobile = layout['mobile']
    axis = 'x2="0" y2="1"' if mobile else ''
    stops = ((0, 1), (.40, 1), (.51, .65), (.66, 0), (1, 0)) if mobile else (
        (0, 1), (.29, .98), (.46, .77), (.59, 0), (1, 0))
    fade = ''.join(f'<stop offset="{p}" stop-color="#050507" stop-opacity="{a}"/>' for p, a in stops)
    safe = f'x="0" y="402" width="{w}" height="{h-402}"' if mobile else f'x="530" y="81" width="{w-530}" height="{h-130}"'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" id="still" role="img" aria-labelledby="title desc">
<!-- Generated by tools/render_cosmos.py -->
<title id="title">Khushal Dangar — Crimson Cosmos</title>
<desc id="desc">Computer Science &amp; Artificial Intelligence. Exploring complexity. Engineering possibilities. A luminous crimson accretion disk bends around a dark black hole.</desc>
{MOTION}
<defs>
  <clipPath id="art-safe"><rect {safe}/></clipPath>
  <linearGradient id="type-shade" {axis}>{fade}</linearGradient>
  <linearGradient id="frame-shade" x2="0" y2="1"><stop stop-color="#050507"/><stop offset=".18" stop-color="#050507" stop-opacity="0"/><stop offset=".79" stop-color="#050507" stop-opacity="0"/><stop offset="1" stop-color="#050507"/></linearGradient>
</defs>
<rect width="{w}" height="{h}" fill="#050507"/>
{picture_payload(img, layout)}
{sky_marks(layout)}
<rect width="{w}" height="{h}" fill="url(#type-shade)"/>
<rect width="{w}" height="{h}" fill="url(#frame-shade)"/>
{typography(layout)}
</svg>
'''


DIVIDER = '''<svg xmlns="http://www.w3.org/2000/svg" width="480" height="28" viewBox="0 0 480 28">
<defs>
  <linearGradient id="rule" gradientUnits="userSpaceOnUse" x1="0" x2="480"><stop stop-color="#929299" stop-opacity="0"/><stop offset=".25" stop-color="#929299" stop-opacity=".52"/><stop offset=".5" stop-color="#929299" stop-opacity=".72"/><stop offset=".75" stop-color="#929299" stop-opacity=".52"/><stop offset="1" stop-color="#929299" stop-opacity="0"/></linearGradient>
</defs>
<path d="M1 14h202m74 0h202" stroke="url(#rule)" fill="none"/>
<g transform="translate(240 14) rotate(-14)" fill="none">
  <ellipse rx="31" ry="6.5" stroke="#929299" stroke-opacity=".72"/>
  <path d="M-31 0c0 4 14 6.5 31 6.5S31 4 31 0" stroke="#DC143C" stroke-opacity=".82"/>
  <circle r="4" fill="#790019" stroke="#DC143C"/>
  <circle cx="22" cy="4.6" r="1.5" fill="#DC143C"/>
</g>
</svg>
'''


def main():
    img = artwork()
    desktop = dict(size=(1280, 534), image=(468, -164, .90), mobile=False,
                   stars_start=585, stars=((1178, 135, True), (760, 163, False), (1155, 409, False), (1012, 108, False)))
    mobile = dict(size=(620, 785), image=(-129, 263, .69), mobile=True,
                  stars_start=0, stars=((520, 454, True), (123, 513, False), (546, 649, False), (186, 668, False)))
    for suffix, layout in (('', desktop), ('-mobile', mobile)):
        (ASSETS / f'crimson-cosmos{suffix}.svg').write_text(cover(img, layout), encoding='utf-8')
    (ASSETS / 'cosmos-divider.svg').write_text(DIVIDER, encoding='utf-8')


if __name__ == '__main__':
    main()
