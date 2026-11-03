// Apply only the two hooks against the pinned upstream source.
package main

import (
	"os"
	"strings"
)

func main() {
	for path, edits := range map[string][][2]string{
		"server_session_media.go": {
			{"type serverSessionMedia struct {", "type serverSessionMedia struct {\n\tearlySenderReport earlySenderReport"},
			{"\t\t\tformat := ssm.findFormatByRemoteSSRC(sr.SSRC)\n\t\t\tif format != nil {\n\t\t\t\tformat.rtpReceiver.ProcessSenderReport(sr, now)\n\t\t\t}", "\t\t\tssm.processSenderReport(sr, now)"},
		},
		"server_session_format.go": {
			{"\tpkts, lost := ssf.rtpReceiver.ProcessPacket2(pkt, now, ssf.format.PTSEqualsDTS(pkt))", "\tssf.ssm.applyEarlySenderReport(ssf, header.SSRC)\n\tpkts, lost := ssf.rtpReceiver.ProcessPacket2(pkt, now, ssf.format.PTSEqualsDTS(pkt))"},
		},
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			panic(err)
		}
		text := string(data)
		for _, edit := range edits {
			if strings.Count(text, edit[0]) != 1 {
				panic("Unexpected pinned source: " + path)
			}
			text = strings.Replace(text, edit[0], edit[1], 1)
		}
		if err := os.WriteFile(path, []byte(text), 0644); err != nil {
			panic(err)
		}
	}
}
