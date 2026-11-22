# V-Count · Shortlink Detector & Counter (Streamlit)

VerifyGN VFGN26 (`models/best.vf`) shortlink detection with live counting and OK / NOK check.

## Features
- **Image inspection**: upload an image, take a camera snapshot, or use a sample image → detect, number and count shortlinks.
- **Live camera counting**
  - 🌐 *Browser camera* (WebRTC): phone or laptop webcam, works on deployed apps (needs HTTPS).
  - 🔌 *Server camera*: USB index (`0`, `1`, …) or IP/RTSP URL on the machine running the app.
  - Stable live count (most frequent count over last N frames) to stop flicker.
- **Set count & check**: set the expected count in the sidebar → shows **OK** (match) or **NOK** (with +/- difference), also drawn on the video.
- **CHECK** button saves the result. The History tab shows OK/NOK totals and yield, and lets you export a CSV report.

## Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
```
Or double-click `run_app.bat`. Open http://localhost:8501. Other devices on your network can use `http://<PC-IP>:8501`.

> Browsers only allow camera access on `localhost` or **HTTPS**. To use a phone camera, deploy behind HTTPS
> (Streamlit Cloud, nginx + certificate, or a tunnel such as cloudflared/ngrok).

## Deploy
**Docker**
```bash
docker build -t vcount .
docker run -p 8501:8501 vcount
```
**Streamlit Community Cloud**: push this folder to GitHub (include `models/best.vf`, `assets/`, `vfgn/`) and set `main.py` as the entry file.

### WebRTC behind strict networks
If the live browser camera stays on "connecting", add a TURN server via `.streamlit/secrets.toml`:
```toml
ICE_SERVERS = '[{"urls":["turn:your.turn.server:3478"],"username":"user","credential":"pass"}]'
```
You can also set it with the `ICE_SERVERS` environment variable.

## Environment variables
| Var | Purpose |
|---|---|
| `VCOUNT_MODEL` | Path to another `.vf` model (default `models/best.vf`) |
| `VCOUNT_MODEL_KEY` | Decryption key if the `.vf` model is encrypted |
| `ICE_SERVERS` | JSON list of STUN/TURN servers for WebRTC |

## Structure
```
main.py          Streamlit UI
engine.py        VFGN26 inference + annotation + OK/NOK logic
vfgn/            VerifyGN .vf model framework
models/best.vf   Shortlink model
assets/          Logo + sample images
```
