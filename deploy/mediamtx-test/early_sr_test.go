package gortsplib

import (
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5/pkg/rtpreceiver"
	"github.com/pion/rtcp"
)

func TestEarlySenderReport(t *testing.T) {
	for _, order := range []string{"before-rtp", "wrong-ssrc", "newer-report"} {
		t.Run(order, func(t *testing.T) {
			rr := &rtpreceiver.Receiver{ClockRate: 90000, LocalSSRC: 1,
				Period: time.Hour, WritePacketRTCP: func(rtcp.Packet) {}}
			if err := rr.Initialize(); err != nil {
				t.Fatal(err)
			}
			defer rr.Close()
			forma := &serverSessionFormat{rtpReceiver: rr}
			media := &serverSessionMedia{formats: map[uint8]*serverSessionFormat{96: forma}}
			now := time.Unix(1000, 0)
			sr := &rtcp.SenderReport{SSRC: 42, NTPTime: uint64(2208989800) << 32, RTPTime: 1000}
			media.processSenderReport(sr, now)
			if _, ok := rr.PacketNTP(1000); ok {
				t.Fatal("unmatched report established a clock")
			}
			if order == "wrong-ssrc" {
				media.applyEarlySenderReport(forma, 99)
				if _, ok := rr.PacketNTP(1000); ok {
					t.Fatal("foreign SSRC established a clock")
				}
			}
			forma.remoteSSRCFilled, forma.remoteSSRCValue = true, 42
			if order == "newer-report" {
				newer := *sr
				newer.NTPTime += uint64(1) << 32
				media.processSenderReport(&newer, now.Add(time.Second))
				now = now.Add(time.Second)
			}
			media.applyEarlySenderReport(forma, 42)
			got, ok := rr.PacketNTP(1000)
			if !ok || !got.Equal(now) || media.earlySenderReport.report != nil {
				t.Fatalf("first RTP clock missing or rewound: %v %v, expected %v", got, ok, now)
			}
		})
	}
}
