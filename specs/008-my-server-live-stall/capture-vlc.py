"""Native RTSP player statistics; displayed includes redraws and is not unique-frame FPS."""
import ctypes as c
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


class Stats(c.Structure):
    _fields_ = [(name, c.c_float if name.startswith('f_') else c.c_int) for name in (
        'i_read_bytes', 'f_input_bitrate', 'i_demux_read_bytes', 'f_demux_bitrate',
        'i_demux_corrupted', 'i_demux_discontinuity', 'i_decoded_video', 'i_decoded_audio',
        'i_displayed_pictures', 'i_lost_pictures', 'i_played_abuffers', 'i_lost_abuffers',
        'i_sent_packets', 'i_sent_bytes', 'f_send_bitrate')]


def capture(directory, endpoint, output):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    before = time.time()
    with opener.open(endpoint, timeout=5) as response:
        campaign = json.load(response)
    received = time.time()
    offset = campaign['now'] - (before + received) / 2
    now = lambda: time.time() + offset
    deadline = campaign['deadline_unix']
    assert 0 < deadline - now() <= 600, 'Use an existing live campaign deadline; never restart its timer'
    dll_directory = os.add_dll_directory(str(directory))
    os.environ['VLC_PLUGIN_PATH'] = str(directory / 'plugins')
    core = c.CDLL(str(directory / 'libvlccore.dll'))
    library = c.CDLL(str(directory / 'libvlc.dll'))
    signatures = {
        'libvlc_new': (c.c_void_p, [c.c_int, c.POINTER(c.c_char_p)]),
        'libvlc_release': (None, [c.c_void_p]),
        'libvlc_get_version': (c.c_char_p, []),
        'libvlc_media_new_location': (c.c_void_p, [c.c_void_p, c.c_char_p]),
        'libvlc_media_release': (None, [c.c_void_p]),
        'libvlc_media_player_new_from_media': (c.c_void_p, [c.c_void_p]),
        'libvlc_media_player_play': (c.c_int, [c.c_void_p]),
        'libvlc_media_player_stop': (None, [c.c_void_p]),
        'libvlc_media_player_release': (None, [c.c_void_p]),
        'libvlc_media_player_get_state': (c.c_int, [c.c_void_p]),
        'libvlc_media_player_get_time': (c.c_int64, [c.c_void_p]),
        'libvlc_media_player_get_rate': (c.c_float, [c.c_void_p]),
        'libvlc_audio_set_mute': (None, [c.c_void_p, c.c_int]),
        'libvlc_media_get_stats': (c.c_int, [c.c_void_p, c.POINTER(Stats)]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(library, name)
        function.restype, function.argtypes = result, args
    options = [b'--ignore-config', b'--intf=dummy', b'--no-video-title-show', b'--no-osd', b'--stats',
               b'--rtsp-tcp', b'--network-caching=300']
    instance = library.libvlc_new(len(options), (c.c_char_p * len(options))(*options))
    assert instance, 'VLC initialization failed'
    ids = ['1aed319f-e680-4f52-ad70-cd9b0be0a220', 'e9d6a3ab-3843-48f4-8083-7dc8f89759b1']
    players = []
    with output.open('x', encoding='utf-8') as log:
        def write(value):
            log.write(json.dumps(dict(unix=now(), monotonic=time.monotonic(), **value)) + '\n')
            log.flush()

        write(dict(kind='start', campaign=campaign, clock_offset_seconds=offset,
                   clock_roundtrip_seconds=received-before, options=[v.decode() for v in options],
                   version=library.libvlc_get_version().decode(),
                   limit='Video displayed counts include redraws. Stats are periodic aggregates, not per-frame presentation times.'))
        def open_player(index, epoch):
            media = library.libvlc_media_new_location(instance, ('rtsp://172.18.20.118:8654/redacted/' + ids[index]).encode())
            assert media
            player = library.libvlc_media_player_new_from_media(media)
            assert player and library.libvlc_media_player_play(player) == 0
            library.libvlc_audio_set_mute(player, 1)
            write(dict(kind='player_opened', stream=index, epoch=epoch))
            return [media, player, epoch, None]

        def close_player(value):
            library.libvlc_media_player_stop(value[1])
            library.libvlc_media_player_release(value[1])
            library.libvlc_media_release(value[0])

        try:
            for index in range(2):
                players.append(open_player(index, 1))
            while now() < deadline:
                rows = []
                for index, value in enumerate(players):
                    media, player, epoch, retry_at = value
                    stats = Stats()
                    valid = library.libvlc_media_get_stats(media, c.byref(stats))
                    state = library.libvlc_media_player_get_state(player)
                    rows.append(dict(state=state, epoch=epoch,
                                     media_ms=library.libvlc_media_player_get_time(player),
                                     rate=library.libvlc_media_player_get_rate(player),
                                     stats={key: getattr(stats, key) for key, _ in stats._fields_} if valid else None))
                    if state in (6, 7):
                        if retry_at is None:
                            value[3] = now() + 1
                        elif now() >= retry_at and now() < deadline:
                            close_player(value)
                            players[index] = open_player(index, epoch + 1)
                write(dict(kind='sample', players=rows))
                time.sleep(max(0, min(.1, deadline-now())))
        finally:
            write(dict(kind='sampling_stopped'))
            for value in players:
                close_player(value)
            library.libvlc_release(instance)
            write(dict(kind='released'))
    dll_directory.close()


if __name__ == '__main__':
    assert c.sizeof(Stats) == 60 and Stats.i_displayed_pictures.offset == 32
    if sys.argv[1:] == ['--self-check']:
        print('VLC stats ABI self-check passed')
    else:
        capture(Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]))
