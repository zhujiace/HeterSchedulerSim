/*
Copy Right. The EHPCL Authors.
*/

#include "segment.h"


bool Segment::resetSegment() {
    executedAt.clear();
    if (segmentRemainLength !=0) return false;
    segmentRemainLength = segmentLength;
    currentProcessor = 999999;
    return true;
}

bool Segment::executeSegment(TimeStamp_t timeStamp) {
    if (segmentRemainLength<=0) return false;
    if (segmentPreemption==SegmentPreemption_t::NONPREEMPTIVE)
    if (!executedAt.empty() && executedAt.back()+1!=timeStamp) return false;
    segmentRemainLength--;
    executedAt.push_back(timeStamp);
    if (segmentRemainLength == 0) {
        segmentCompleted = true;
        segmentReady = false;
    }
    return true;
}

