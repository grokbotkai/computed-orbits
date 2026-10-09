"""Write short-01.srt from the same caption chunks burned into the video (render.CAPS)."""
import os, render
def ts(s): h=int(s//3600); m=int(s%3600//60); sec=s%60; return f"{h:02d}:{m:02d}:{int(sec):02d},{int(round((sec-int(sec))*1000)):03d}"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "short-01.srt"), "w") as f:
    for i, (a, b, txt) in enumerate(render.CAPS, 1):
        f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{txt.replace(chr(10), ' ')}\n\n")
