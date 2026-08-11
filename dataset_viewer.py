#!/usr/bin/env python3
"""Serve a lightweight dataset video viewer from root listing txt files.

Usage:
  python dataset_viewer.py
  python dataset_viewer.py contphy.txt
  python dataset_viewer.py --port 8899
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, urlparse
import webbrowser


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Viewer</title>
  <style>
    :root {
      --bg: #0f1218;
      --panel: #171b24;
      --panel-2: #1f2531;
      --fg: #e7edf8;
      --muted: #9eabc1;
      --accent: #4fc3f7;
      --border: #2d3647;
      --selected: #253247;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
      color: var(--fg);
      background: radial-gradient(circle at 20% -10%, #1a2230 0%, var(--bg) 40%);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      background: color-mix(in hsl, var(--bg) 82%, black);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
      padding: 12px 16px;
    }
    .title {
      margin: 0;
      font-size: 1rem;
      line-height: 1.3;
      font-weight: 700;
    }
    .sub {
      margin: 4px 0 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .dataset-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .dataset-list a {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--fg);
      text-decoration: none;
      background: #121825;
      font-size: 0.82rem;
      line-height: 1.2;
    }
    .dataset-list a:hover {
      border-color: var(--accent);
      color: #f4f8ff;
    }
    .dataset-list a.selected {
      background: var(--selected);
      border-color: #5c7395;
      font-weight: 700;
    }
    main {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 12px;
    }
    .card {
      background: linear-gradient(180deg, var(--panel-2), var(--panel));
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: clip;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    }
    .video-wrap {
      position: relative;
      aspect-ratio: 16 / 9;
      background: #0b0e14;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #0b0e14;
    }
    .badge {
      position: absolute;
      left: 8px;
      top: 8px;
      font-size: 0.72rem;
      color: #d7e3fb;
      border: 1px solid #355070;
      background: rgba(24, 35, 53, 0.75);
      padding: 3px 7px;
      border-radius: 999px;
      letter-spacing: 0.02em;
    }
    .meta {
      padding: 8px 10px 10px;
      border-top: 1px solid var(--border);
    }
    .name {
      margin: 0;
      font-size: 0.87rem;
      font-weight: 600;
      color: #dce8ff;
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .path {
      margin: 4px 0 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.76rem;
      color: var(--muted);
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .details {
      margin: 6px 0 0;
      font-size: 0.76rem;
      color: #b7d4ff;
      line-height: 1.35;
    }
    .empty {
      margin: 12px;
      padding: 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #121827;
      color: var(--muted);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <header>
    <h1 class="title">Dataset Viewer</h1>
    <p class="sub">{count} videos from {listing_name}</p>
    <ul class="dataset-list">
      {dataset_links}
    </ul>
  </header>
  {body}
  <script>
    const pendingTimers = new Map();
    const inViewport = new Map();

    function ensureLoaded(video) {
      if (!video.src) {
        video.src = video.dataset.src;
        video.load();
      }
    }

    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const video = entry.target;

        if (entry.isIntersecting) {
          inViewport.set(video, true);
          if (!pendingTimers.has(video) && !video.src) {
            const timerId = setTimeout(() => {
              const stillVisible = inViewport.get(video) === true;
              if (stillVisible) {
                ensureLoaded(video);
                observer.unobserve(video);
              }
              pendingTimers.delete(video);
            }, 200);
            pendingTimers.set(video, timerId);
          }
        } else {
          inViewport.set(video, false);
          const timerId = pendingTimers.get(video);
          if (timerId) {
            clearTimeout(timerId);
            pendingTimers.delete(video);
          }
        }
      }
    }, {
      root: null,
      rootMargin: '0px',
      threshold: 0.01,
    });

    document.querySelectorAll('video[data-src]').forEach((video) => {
      observer.observe(video);

      // Hover-to-play, paused by default.
      video.addEventListener('mouseenter', async () => {
        ensureLoaded(video);
        try {
          await video.play();
        } catch (_) {
          // Ignore play interruptions/policy issues.
        }
      });

      // Pause if user tabs away.
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) video.pause();
      });
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class VideoMeta:
    duration_s: str
    resolution: str
    fps: str
    codec: str
    frames: str


class ViewerState:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.listings: dict[str, Path] = scan_root_txt_files(root_dir)
        self.video_cache: dict[str, list[Path]] = {}
        self.meta_cache: dict[Path, VideoMeta | None] = {}
        self.ffprobe_path = shutil.which("ffprobe")

    def available_datasets(self) -> list[str]:
        return sorted(self.listings.keys(), key=lambda x: x.lower())

    def get_listing_for_dataset(self, dataset: str) -> Path | None:
        return self.listings.get(dataset)

    def get_video_paths(self, dataset: str) -> list[Path]:
        if dataset in self.video_cache:
            return self.video_cache[dataset]

        listing_file = self.get_listing_for_dataset(dataset)
        if listing_file is None:
            return []

        videos = parse_listing(listing_file, self.root_dir)
        self.video_cache[dataset] = videos
        return videos

    def get_meta(self, path: Path) -> VideoMeta | None:
        if path in self.meta_cache:
            return self.meta_cache[path]

        meta = probe_video(path, self.ffprobe_path)
        self.meta_cache[path] = meta
        return meta


def scan_root_txt_files(root_dir: Path) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for txt_file in sorted(root_dir.glob("*.txt")):
        datasets[txt_file.name] = txt_file.resolve()
    return datasets


def parse_listing(listing_file: Path, base_dir: Path) -> list[Path]:
    video_paths: list[Path] = []
    with listing_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            candidate = Path(line)
            if candidate.is_absolute():
                resolved = candidate
            else:
                resolved = (base_dir / candidate).resolve()
                if not resolved.exists():
                    resolved = (listing_file.parent / candidate).resolve()

            if resolved.exists() and resolved.is_file():
                video_paths.append(resolved)

    return video_paths


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            den = float(right)
            if den == 0:
                return None
            return float(left) / den
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(path: Path, ffprobe_path: str | None) -> VideoMeta | None:
    if ffprobe_path is None:
        return None

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    stream0 = streams[0] if streams else {}
    fmt = payload.get("format") or {}

    width = stream0.get("width")
    height = stream0.get("height")
    resolution = "n/a" if width is None or height is None else f"{width}x{height}"

    fps_value = parse_fraction(stream0.get("avg_frame_rate"))
    if fps_value is None:
        fps_value = parse_fraction(stream0.get("r_frame_rate"))
    fps = "n/a" if fps_value is None else f"{fps_value:.3f}".rstrip("0").rstrip(".")

    codec_name = stream0.get("codec_name")
    codec = str(codec_name) if codec_name else "n/a"

    frames_raw = stream0.get("nb_frames")
    frames = str(frames_raw) if frames_raw not in (None, "") else "n/a"

    duration_raw = fmt.get("duration")
    try:
        duration = "n/a" if duration_raw is None else f"{float(duration_raw):.2f}s"
    except (TypeError, ValueError):
        duration = "n/a"

    return VideoMeta(
        duration_s=duration,
        resolution=resolution,
        fps=fps,
        codec=codec,
        frames=frames,
    )


def iter_dataset_links(dataset_names: Iterable[str], selected: str) -> str:
    links: list[str] = []
    for name in dataset_names:
        css_class = "selected" if name == selected else ""
        url = f"/?dataset={quote(name)}"
        links.append(
            f'<li><a class="{css_class}" href="{url}">{html.escape(name)}</a></li>'
        )
    return "\n".join(links)


def iter_cards(
    dataset_name: str,
    video_paths: Iterable[Path],
    base_dir: Path,
    state: ViewerState,
) -> str:
    cards: list[str] = []
    for idx, path in enumerate(video_paths):
        try:
            short = path.relative_to(base_dir)
            display_path = str(short)
        except ValueError:
            display_path = str(path)

        details_html = ""
        if (idx) % 100 == 0:
            meta = state.get_meta(path)
            if meta is None:
                details_text = "Duration: n/a | Resolution: n/a | FPS: n/a | Codec: n/a | Frames: n/a"
            else:
                details_text = (
                    f"Duration: {meta.duration_s} | "
                    f"Resolution: {meta.resolution} | "
                    f"FPS: {meta.fps} | "
                    f"Codec: {meta.codec} | "
                    f"Frames: {meta.frames}"
                )
            details_html = f'<p class="details">{html.escape(details_text)}</p>'

        data_src = f"/video?dataset={quote(dataset_name)}&i={idx}"
        cards.append(
            """
            <article class="card">
              <div class="video-wrap">
                <span class="badge">#{num}</span>
                <video
                  preload="none"
                  muted
                  playsinline
                  controls
                  data-src="{data_src}"
                ></video>
              </div>
              <div class="meta">
                <p class="name">{name}</p>
                <p class="path">{path}</p>
                {details}
              </div>
            </article>
            """.format(
                num=idx + 1,
                data_src=html.escape(data_src),
                name=html.escape(path.name),
                path=html.escape(display_path),
                details=details_html,
            )
        )

    return "\n".join(cards)


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "application/octet-stream"


def parse_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    if not range_header.startswith("bytes="):
        return None

    value = range_header.split("=", 1)[1].strip()
    if "," in value:
        return None

    start_s, end_s = value.split("-", 1)
    if not start_s and not end_s:
        return None

    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    else:
        length = int(end_s)
        if length <= 0:
            return None
        start = max(0, file_size - length)
        end = file_size - 1

    if start < 0 or end < start or start >= file_size:
        return None

    end = min(end, file_size - 1)
    return start, end


def build_page(state: ViewerState, selected_dataset: str) -> str:
    dataset_names = state.available_datasets()
    listing_file = state.get_listing_for_dataset(selected_dataset)
    video_paths = state.get_video_paths(selected_dataset)

    if listing_file is None:
        listing_name = "(dataset not found)"
        body = '<div class="empty">Selected dataset was not found.</div>'
        count = 0
    elif not video_paths:
        listing_name = listing_file.name
        body = '<div class="empty">No valid video files found in this listing.</div>'
        count = 0
    else:
        listing_name = listing_file.name
        cards = iter_cards(selected_dataset, video_paths, state.root_dir, state)
        body = f"<main>{cards}</main>"
        count = len(video_paths)

    page = HTML_TEMPLATE
    page = page.replace("{count}", str(count))
    page = page.replace("{listing_name}", html.escape(listing_name))
    page = page.replace("{dataset_links}", iter_dataset_links(dataset_names, selected_dataset))
    page = page.replace("{body}", body)
    return page


def make_handler(state: ViewerState, default_dataset: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DatasetViewer/0.2"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                params = parse_qs(parsed.query)
                selected = params.get("dataset", [default_dataset])[0]
                if selected not in state.listings:
                    selected = default_dataset
                self._send_html(build_page(state, selected))
                return

            if parsed.path == "/video":
                params = parse_qs(parsed.query)
                dataset_values = params.get("dataset")
                idx_values = params.get("i")
                if not dataset_values or not idx_values:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing dataset or video index")
                    return

                dataset = dataset_values[0]
                if dataset not in state.listings:
                    self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
                    return

                try:
                    idx = int(idx_values[0])
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video index")
                    return

                videos = state.get_video_paths(dataset)
                if idx < 0 or idx >= len(videos):
                    self.send_error(HTTPStatus.NOT_FOUND, "Video index out of range")
                    return

                self._send_video(videos[idx])
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_html(self, body: str) -> None:
            content = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_video(self, path: Path) -> None:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return

            content_type = content_type_for(path)
            range_header = self.headers.get("Range")
            byte_range = parse_range(range_header, size) if range_header else None

            if byte_range is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with path.open("rb") as f:
                    self._copy_file(f, self.wfile)
                return

            start, end = byte_range
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        @staticmethod
        def _copy_file(src, dst) -> None:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)

    return Handler


def choose_default_dataset(state: ViewerState, requested: str | None) -> str:
    names = state.available_datasets()
    if not names:
        raise SystemExit("No root .txt files found in current directory.")

    if requested is None:
        return names[0]

    req_name = Path(requested).name
    if req_name in state.listings:
        return req_name

    if requested in state.listings:
        return requested

    return names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a local dataset video viewer")
    parser.add_argument(
        "listing",
        nargs="?",
        help="Optional initial listing file name (e.g., contphy.txt)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8890, help="Port to bind (default: 8890)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    root_dir = Path.cwd().resolve()
    state = ViewerState(root_dir)
    default_dataset = choose_default_dataset(state, args.listing)

    handler = make_handler(state, default_dataset)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/?dataset={quote(default_dataset)}"

    print(f"Found {len(state.listings)} root txt files.")
    print("Available datasets:")
    for name in state.available_datasets():
        print(f"  - {name}")
    print(f"Default dataset: {default_dataset}")
    print(f"Viewer URL: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
