package gortsplib

import (
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5/pkg/format"
)

func TestDelayedMediaRTPInfo(t *testing.T) {
	now := time.Unix(1000, 0)
	for _, age := range []time.Duration{0, 50 * time.Millisecond, 2 * time.Second, -2 * time.Second} {
		ssf := &serverStreamFormat{
			format: &format.H264{PayloadTyp: 96}, firstSent: true,
			lastRTP: 100000, lastNTP: now.Add(-age), lastSequenceNumber: 20,
		}
		entry := ssf.rtpInfoEntry(now)
		if age > time.Second || age < -time.Second {
			if entry != nil {
				t.Fatalf("delayed media advertised an extrapolated RTP timestamp: %v", entry)
			}
		} else if entry == nil || entry.Timestamp == nil || *entry.SequenceNumber != 21 {
			t.Fatalf("fresh media lost its existing RTP-Info: %v", entry)
		}
	}
}
