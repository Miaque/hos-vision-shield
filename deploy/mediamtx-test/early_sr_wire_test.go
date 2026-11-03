package gortsplib

import (
	"bufio"
	"net"
	"testing"
	"time"

	"github.com/bluenviron/gortsplib/v5/pkg/base"
	"github.com/bluenviron/gortsplib/v5/pkg/conn"
	"github.com/bluenviron/gortsplib/v5/pkg/description"
	"github.com/bluenviron/gortsplib/v5/pkg/format"
	"github.com/bluenviron/gortsplib/v5/pkg/headers"
	"github.com/bluenviron/gortsplib/v5/pkg/ntp"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/stretchr/testify/require"
)

func TestEarlySenderReportTCP(t *testing.T) {
	type observed struct {
		at time.Time
		ok bool
	}
	received := make(chan observed, 3)
	s := &Server{RTSPAddress: "127.0.0.1:8554", Handler: &testServerHandler{
		onAnnounce: func(*ServerHandlerOnAnnounceCtx) (*base.Response, error) {
			return &base.Response{StatusCode: base.StatusOK}, nil
		},
		onSetup: func(*ServerHandlerOnSetupCtx) (*base.Response, *ServerStream, error) {
			return &base.Response{StatusCode: base.StatusOK}, nil, nil
		},
		onRecord: func(ctx *ServerHandlerOnRecordCtx) (*base.Response, error) {
			ctx.Session.OnPacketRTPAny(func(m *description.Media, _ format.Format, p *rtp.Packet) {
				at, ok := ctx.Session.PacketNTP(m, p)
				received <- observed{at, ok}
			})
			return &base.Response{StatusCode: base.StatusOK}, nil
		},
	}}
	require.NoError(t, s.Start())
	defer s.Close()
	for attempt := 0; attempt < 3; attempt++ {
		socket, err := net.Dial("tcp", "127.0.0.1:8554")
		require.NoError(t, err)
		c := conn.NewConn(bufio.NewReader(socket), socket)
		u := "rtsp://127.0.0.1:8554/early-sr"
		doAnnounce(t, c, u, []*description.Media{testH264Media})
		res, _ := doSetup(t, c, u+"/"+testH264Media.Control, &headers.Transport{
			Protocol: headers.TransportProtocolTCP, InterleavedIDs: &[2]int{0, 1},
			Delivery: new(headers.TransportDeliveryUnicast), Mode: new(headers.TransportModeRecord),
		}, "")
		doRecord(t, c, u, readSession(t, res))
		at := time.Unix(1000+int64(attempt), 0)
		buf := make([]byte, 2048)
		require.NoError(t, c.WriteInterleavedFrame(&base.InterleavedFrame{Channel: 1,
			Payload: mustMarshalPacketRTCP(&rtcp.SenderReport{SSRC: 42, NTPTime: ntp.Encode(at), RTPTime: 90000})}, buf))
		require.NoError(t, c.WriteInterleavedFrame(&base.InterleavedFrame{Channel: 0,
			Payload: mustMarshalPacketRTP(&rtp.Packet{Header: rtp.Header{Version: 2, PayloadType: 96,
				SSRC: 42, SequenceNumber: 1, Timestamp: 90000, Marker: true}, Payload: []byte{0x65, 1, 2}})}, buf))
		select {
		case got := <-received:
			require.True(t, got.ok, "first RTP lost the preceding sender report")
			require.True(t, got.at.Equal(at))
		case <-time.After(3 * time.Second):
			t.Fatal("first RTP did not arrive")
		}
		socket.Close()
	}
}
