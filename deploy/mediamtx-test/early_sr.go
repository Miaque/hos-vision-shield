package gortsplib

import (
	"sync"
	"time"

	"github.com/pion/rtcp"
)

type earlySenderReport struct {
	sync.Mutex
	report *rtcp.SenderReport
	at     time.Time
}

func (ssm *serverSessionMedia) processSenderReport(sr *rtcp.SenderReport, now time.Time) {
	pending := &ssm.earlySenderReport
	pending.Lock()
	defer pending.Unlock()
	if forma := ssm.findFormatByRemoteSSRC(sr.SSRC); forma != nil {
		forma.rtpReceiver.ProcessSenderReport(sr, now)
		if pending.report != nil && pending.report.SSRC == sr.SSRC {
			pending.report = nil
		}
		return
	}
	// One bounded slot: bind an early report only after matching the actual RTP SSRC.
	pending.report, pending.at = sr, now
}

func (ssm *serverSessionMedia) applyEarlySenderReport(forma *serverSessionFormat, ssrc uint32) {
	pending := &ssm.earlySenderReport
	pending.Lock()
	defer pending.Unlock()
	if pending.report != nil && pending.report.SSRC == ssrc {
		forma.rtpReceiver.ProcessSenderReport(pending.report, pending.at)
		pending.report = nil
	}
}
