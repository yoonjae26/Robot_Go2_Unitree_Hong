#!/usr/bin/env python3
"""
Plays audio through the Go2's own speaker over the shared WebRTC data channel,
instead of the local machine's speaker -- uses the audiohub RPCs vendored in
Test/unitree_webrtc_connect/unitree_webrtc_connect/webrtc_audiohub.py, the same
way webrtc_sport_client.py reuses the SPORT_MOD RPCs: same shared connection,
same asyncio.run_coroutine_threadsafe bridge from a synchronous caller thread
(see WebRTCSportClient's docstring for why a *second* connection isn't opened).

"Megaphone" mode (enter -> upload -> exit) is the ephemeral/live playback path,
as opposed to upload_audio_file()+play_by_uuid(), which permanently stores the
clip as a named library item on the robot -- unsuitable here since every TTS
reply is different text and would otherwise pile up as one library entry per
utterance forever.

Confirmed working end-to-end on real hardware: enter_megaphone acks, and
every upload chunk comes back with {"status":{"code":0}}. The vendored
reference's chunk size (4096 base64 chars) + 0.1s pacing delay between
chunks meant a ~13s TTS sentence took ~380 chunks * 0.1s+RTT =~ 78s just to
transfer -- observed live, and slow enough that the whole dashboard had to
be killed rather than waited out.

That pacing is redundant, not a safety margin: each chunk already blocks on
its own ack (see _call()) before the next one is sent, so the channel is
never given more than one in-flight request regardless of any extra sleep --
correctness/ordering comes from waiting for that ack, not from a fixed
delay. So chunk size was raised 4x (4096 -> 16384) to cut the number of
round trips, and the delay cut 5x (0.1s -> 0.02s) as a small residual buffer
rather than removed outright, since neither has been verified against a
robot-side message-size ceiling.

Even at 16384, a long (paragraph-length) TTS reply -- e.g. the vision-
analysis feature's scene descriptions -- still took ~30s in practice: with
per-chunk RTT now the dominant cost (not the sleep), total time scales
with chunk *count*, so long text is still slow regardless of pacing. Chunk
size was raised again (16384 -> 32768) on the strength of 16384 having come
back clean, but that *did* turn out to be past this robot's real ceiling --
audio came back audibly distorted/crackling ("rè") at 32768, reported live
after a vision-analysis reply. Reverted to 16384 (the last confirmed-clean
value) rather than searched for the exact breakpoint in between, since
correctness matters more here than shaving a few more seconds. The other,
safer lever for long replies is the text itself: dashboard_server.py's
vision-analysis prompt was also tightened to cap response length, since
half the fix for "long reply is slow to transfer" is not generating a long
reply in the first place.

Timeouts are applied *per RPC call* (enter/each chunk/exit) rather than one
ceiling over the whole upload -- a single lump timeout would either be too
short for a long sentence (cutting off a transfer that was working fine) or
too long for a genuinely stuck call.

enter()/exit() are split out from playback (play_wav() vs. play_in_session())
so a caller can enter megaphone mode ONCE and play several clips back-to-back
in one session. dashboard_server.py's _speak() briefly used this to play a
long reply sentence-by-sentence (so the robot would start talking after the
first, much shorter, sentence's transfer instead of waiting for the whole
reply) but was reverted back to sending the whole reply as a single
play_wav() clip at the user's request -- kept simpler on purpose rather than
for a correctness reason found in testing. The split methods stay here
(play_wav() is built out of them) in case multi-clip playback is wanted
again later.
"""

import asyncio
import base64
import concurrent.futures
import logging
import time
import wave
from typing import Optional

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 16384
_CHUNK_DELAY_S = 0.02
_CALL_TIMEOUT_S = 5.0  # per RPC (enter/each chunk/exit) -- a single one hanging is what we want to catch fast


class WebRTCAudioHub:
    def __init__(self, shared_conn, shared_loop: asyncio.AbstractEventLoop):
        self._conn = shared_conn
        self._loop = shared_loop

    def enter(self):
        """Switch the robot into megaphone mode. Call once before one or
        more play_in_session() calls; pair with exit()."""
        self._call(self._enter_megaphone(), what="enter_megaphone")

    def exit(self):
        """Leave megaphone mode. Always call after enter(), even if a
        play_in_session() call in between failed."""
        self._call(self._exit_megaphone(), what="exit_megaphone")

    def play_in_session(self, wav_path: str):
        """Upload and play one clip within an already-open megaphone session
        (see enter()). Blocks until the clip's own duration has elapsed --
        for a long clip this can legitimately take tens of seconds (see
        module docstring), that's not this method hanging, it's the upload.
        Does NOT enter/exit -- callers wanting a single clip should use
        play_wav() instead."""
        with wave.open(wav_path, "rb") as wf:
            duration_s = wf.getnframes() / float(wf.getframerate())
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        self._upload_megaphone(wav_bytes)
        time.sleep(duration_s)  # chunks are sent immediately; don't cut playback off early

    def play_wav(self, wav_path: str):
        """Play a single WAV file through the robot's speaker: enter megaphone
        mode, play it, exit. For playing several clips back-to-back (e.g.
        sentence-by-sentence TTS), call enter() once and play_in_session()
        per clip instead -- see module docstring."""
        self.enter()
        try:
            self.play_in_session(wav_path)
        finally:
            try:
                self.exit()
            except Exception as e:
                logger.warning(f"Robot speaker: exit_megaphone failed: {e}")

    def _call(self, coro, what: str):
        """Run one RPC coroutine on the shared connection's event loop from
        this (synchronous) calling thread, with its own short timeout."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=_CALL_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            # concurrent.futures.TimeoutError stringifies to '' (and, on
            # Python < 3.11, isn't the same class as the builtin TimeoutError
            # -- must be caught by its concurrent.futures name explicitly) --
            # name the step so the fallback log/UI message is informative
            # instead of "failed ()".
            raise TimeoutError(f"{what}: no response from robot within {_CALL_TIMEOUT_S:.0f}s")

    async def _request(self, api_id: int, parameter: Optional[dict] = None):
        from unitree_webrtc_connect.constants import RTC_TOPIC
        payload = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter
        return await self._conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["AUDIO_HUB_REQ"], payload
        )

    async def _enter_megaphone(self):
        from unitree_webrtc_connect.constants import AUDIO_API
        return await self._request(AUDIO_API["ENTER_MEGAPHONE"])

    async def _exit_megaphone(self):
        from unitree_webrtc_connect.constants import AUDIO_API
        return await self._request(AUDIO_API["EXIT_MEGAPHONE"])

    def _upload_megaphone(self, wav_bytes: bytes):
        from unitree_webrtc_connect.constants import AUDIO_API
        b64_data = base64.b64encode(wav_bytes).decode("utf-8")
        chunks = [b64_data[i:i + _CHUNK_SIZE] for i in range(0, len(b64_data), _CHUNK_SIZE)]
        total = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            parameter = {
                "current_block_size": len(chunk),
                "block_content": chunk,
                "current_block_index": i,
                "total_block_number": total,
            }
            self._call(self._request(AUDIO_API["UPLOAD_MEGAPHONE"], parameter), what=f"upload chunk {i}/{total}")
            time.sleep(_CHUNK_DELAY_S)
