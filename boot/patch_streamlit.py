"""Brand Streamlit's loading screen: the Global Market Analyst globe instead of a black page
and the running-man status icon.

Streamlit serves a fixed index.html before the app connects, so the loader has to live in
that file. This copies the globe next to it and inserts the loader once (idempotent); the
deploy runs it after installing the requirements. The loader fades out when the first panel
of the app has rendered, or after 25 s at the latest.
"""
import os
import shutil

import streamlit

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(os.path.dirname(streamlit.__file__), "static")
INDEX = os.path.join(STATIC, "index.html")
MARK = "<!-- gma-boot -->"

HEAD = """<!-- gma-boot -->
    <title>Market Analyst · Global Market Analyst</title>
    <style>
      /* the running-man status icon in the corner */
      [data-testid="stStatusWidget"] { display: none !important; }
      #gma-boot { position: fixed; inset: 0; z-index: 100000; background: #04060b;
        display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 28px;
        transition: opacity .7s ease, visibility .7s; font-family: "Source Sans", system-ui, sans-serif; }
      #gma-boot.gone { opacity: 0; visibility: hidden; }
      #gma-boot .orb { position: relative; width: 240px; height: 240px; }
      #gma-boot .orb img { width: 100%; height: 100%; border-radius: 50%;
        animation: gma-float 4s ease-in-out infinite, gma-glow 2.6s ease-in-out infinite; }
      #gma-boot .orb::before { content: ""; position: absolute; inset: -18px; border-radius: 50%;
        border: 2px solid rgba(34,197,94,.12); border-top-color: #22c55e; border-right-color: rgba(34,197,94,.5);
        animation: gma-spin 1.6s linear infinite; }
      #gma-boot .orb::after { content: ""; position: absolute; inset: -34px; border-radius: 50%;
        border: 1px dashed rgba(96,165,250,.25); animation: gma-spin 9s linear infinite reverse; }
      #gma-boot .name { color: #fff; font-size: 26px; font-weight: 600; letter-spacing: .01em; }
      #gma-boot .sub { color: #8a93a3; font-size: 14px; letter-spacing: .18em; text-transform: uppercase; }
      #gma-boot .sub span { animation: gma-dot 1.4s infinite; opacity: 0; }
      #gma-boot .sub span:nth-child(2) { animation-delay: .2s; }
      #gma-boot .sub span:nth-child(3) { animation-delay: .4s; }
      @keyframes gma-spin { to { transform: rotate(360deg); } }
      @keyframes gma-float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
      @keyframes gma-glow { 0%,100% { box-shadow: 0 0 40px 4px rgba(59,130,246,.25); }
                            50% { box-shadow: 0 0 70px 14px rgba(34,197,94,.30); } }
      @keyframes gma-dot { 0%,100% { opacity: 0; } 50% { opacity: 1; } }
      @media (max-width: 480px) { #gma-boot .orb { width: 180px; height: 180px; } #gma-boot .name { font-size: 22px; } }
      @media (prefers-reduced-motion: reduce) { #gma-boot * { animation: none !important; } }
    </style>"""

BODY = """<!-- gma-boot -->
    <div id="gma-boot" aria-label="Loading Market Analyst">
      <div class="orb"><img src="./gma_globe.jpg" alt=""></div>
      <div class="name">Global Market Analyst</div>
      <div class="sub">Loading live markets<span>.</span><span>.</span><span>.</span></div>
    </div>
    <script>
      (function () {
        var boot = document.getElementById("gma-boot"), t0 = Date.now();
        function ready() {
          // the first data panel of Global Signals, the tab everyone lands on
          var hs = document.querySelectorAll("h1, h2, h3");
          for (var i = 0; i < hs.length; i++) if (/Sector heatmap|Biggest movers/.test(hs[i].textContent)) return true;
          return false;
        }
        var iv = setInterval(function () {
          if (ready() || Date.now() - t0 > 25000) {
            clearInterval(iv); boot.classList.add("gone");
            setTimeout(function () { boot.remove(); }, 900);
          }
        }, 250);
      })();
    </script>"""


def _strip(html: str) -> str:
    """Remove a previously inserted loader (both blocks) so the current one can go in."""
    while MARK in html:
        start = html.index(MARK)
        end_head = html.find("</style>", start)
        end_body = html.find("</script>", start)
        if 0 <= end_head < (end_body if end_body >= 0 else 10 ** 9) and "<style>" in html[start:end_head]:
            html = html[:start] + html[end_head + len("</style>"):]
        else:
            html = html[:start] + html[end_body + len("</script>"):]
    return html


def main() -> None:
    shutil.copy(os.path.join(HERE, "gma_globe.jpg"), os.path.join(STATIC, "gma_globe.jpg"))
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    html = _strip(html)                        # an older version of the loader, if any
    html = html.replace("<title>Streamlit</title>", "", 1)
    html = html.replace("</head>", HEAD + "\n  </head>", 1)
    html = html.replace('<div id="root"></div>', '<div id="root"></div>\n    ' + BODY, 1)
    if BODY not in html:                       # a future index.html without that exact root div
        html = html.replace("</body>", BODY + "\n  </body>", 1)
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    print("loader added to", INDEX)


if __name__ == "__main__":
    main()
