"""
scorechat_app.py
----------------
Streamlit chat interface for ScoreChat.
Requires a populated pgvector database (run ingest_scores.py first).

Usage:
    streamlit run scorechat_app.py
"""

import csv
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from pipeline.chat import chat


def render_score_slice_in_streamlit(musicxml_slice: str, start: int, end: int):
    """Render a MusicXML slice using Verovio WASM inside a Streamlit HTML component."""
    if not musicxml_slice:
        return
    escaped_slice = musicxml_slice.replace("`", "\\`").replace("$", "\\$")
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js" defer></script>
      <style>
        body {{
          background-color: #ffffff;
          margin: 0;
          padding: 8px;
          display: flex;
          justify-content: center;
          align-items: center;
        }}
        #score-container {{
          width: 100%;
          height: auto;
          box-shadow: 0 2px 8px rgba(0,0,0,0.15);
          border-radius: 6px;
          padding: 10px;
          background: #ffffff;
        }}
        .highlighted-measure {{
          fill: rgba(20, 184, 166, 0.15) !important;
          stroke: #14b8a6 !important;
          stroke-width: 3px !important;
        }}
      </style>
    </head>
    <body>
      <div id="score-container" style="color: #6b7280; font-family: sans-serif; font-size: 0.85rem;">Loading score notation...</div>
      <script>
        let toolkit = null;
        document.addEventListener("DOMContentLoaded", () => {{
          if (typeof verovio !== "undefined") {{
            verovio.module.onRuntimeInitialized = () => {{
              toolkit = new verovio.toolkit();
              render();
            }};
          }}
        }});
        
        function render() {{
          if (!toolkit) return;
          const slice = `{escaped_slice}`;
          toolkit.setOptions({{
            pageWidth: 1200,
            pageHeight: 1600,
            scale: 38,
            adjustPageHeight: true,
            select: ["{start}-{end}"]
          }});
          const ok = toolkit.loadData(slice);
          if (ok) {{
            const svg = toolkit.renderToSVG(1);
            const container = document.getElementById("score-container");
            container.innerHTML = svg;
            
            // Highlight measures
            for (let m = {start}; m <= {end}; m++) {{
              const el = document.querySelector("#measure-" + m);
              if (el) {{
                const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                const bbox = el.getBBox();
                rect.setAttribute("x", bbox.x - 30);
                rect.setAttribute("y", bbox.y - 30);
                rect.setAttribute("width", bbox.width + 60);
                rect.setAttribute("height", bbox.height + 60);
                rect.setAttribute("class", "highlighted-measure");
                rect.setAttribute("rx", "8");
                el.parentNode.insertBefore(rect, el);
              }}
            }}
          }} else {{
            document.getElementById("score-container").textContent = "Failed to render score.";
          }}
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=360, scrolling=True)

load_dotenv()

FRSM_CSV = Path("data/frsm_scores.csv")


@st.cache_data(show_spinner=False)
def load_frsm_scores() -> dict[str, list[dict]]:
    """Load FRSM score list from CSV; return dict keyed by composer."""
    if not FRSM_CSV.exists():
        return {}
    scores: dict[str, list[dict]] = defaultdict(list)
    with open(FRSM_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["composer"]].append(row)
    return dict(scores)


@st.cache_resource(show_spinner="Connecting to pipeline...")
def _warmup():
    return chat


def main():
    st.set_page_config(page_title="ScoreChat", page_icon="\U0001f3bc", layout="wide")
    st.title("\U0001f3bc ScoreChat")
    st.caption("Classical Score RAG — ask questions grounded in your uploaded scores.")

    with st.sidebar:
        st.header("FRSM Repertoire")
        frsm = load_frsm_scores()
        if frsm:
            for composer in sorted(frsm.keys()):
                with st.expander(composer):
                    for row in frsm[composer]:
                        label = row["work"]
                        if row.get("nickname"):
                            label += f" “{row['nickname']}”"
                        if row.get("opus"):
                            label += f", Op. {row['opus'].lstrip('Op. ')}"
                        elif row.get("catalog"):
                            label += f", {row['catalog']}"
                        if row.get("imslp_url"):
                            st.markdown(f"- [{label}]({row['imslp_url']})")
                        else:
                            st.markdown(f"- {label}")
        else:
            st.caption(
                "No FRSM CSV found. "
                "Run `python download_scores.py --csv data/frsm_scores.csv`."
            )

    _warmup()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about a piece, key, form, or bar number...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            result = chat(query)

        answer   = result["answer"]
        segments = result.get("segments", [])

        st.markdown(answer)

        if segments:
            with st.expander("Score excerpts"):
                for seg in segments:
                    st.markdown(
                        f"**{seg.get('composer','')} — {seg.get('title','')} "
                        f"({seg.get('opus','')})  "
                        f"mm. {seg.get('measure_start','')}–{seg.get('measure_end','')}**"
                    )
                    st.write(seg.get("summary_text", ""))
                    if seg.get("musicxml_slice"):
                        render_score_slice_in_streamlit(
                            seg["musicxml_slice"],
                            seg.get("measure_start"),
                            seg.get("measure_end"),
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
