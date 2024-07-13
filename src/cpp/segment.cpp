/*
Copy Right. The EHPCL Authors.
*/

#include "segment.h"


bool Segment::resetSegment(bool enforce) {
    if (!enforce && segmentRemainLength !=0) return false;
    executedAt.clear();
    segmentRemainLength = segmentLength;
    currentProcessor = 999999;
    segmentCompleted = false;
    segmentReady = false;
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

