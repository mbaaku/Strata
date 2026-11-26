// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
/// `entries` (#1980): the length of `d_res`, `mirror_table` and `usage` (n_layers x n_expert).  A call whose residency
/// slice lies outside `d_res` - a later layer-split stage's own table - gets no mirror and no counts (a plain miss): the
/// slice used to be taken as an offset from `d_res` anyway and read another allocation (xe page fault on the second card).
/// 0: unchecked, as before.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage = nullptr,
                              long long entries = 0);

/// STRATA_MIRROR_STAGE (opt-in): a device-built plan's groups whose blob address lies outside the VRAM expert
/// arena [arena_lo, arena_hi) are the experts read from the pinned host mirror.  The expert kernels read those with
/// narrow scattered loads, which cross PCIe far below the link's rate (measured on an Arc Pro B70: 26-40 GB/s with
/// 4-byte loads against 57.6 GB/s with 16-byte loads).  This copies the first `cap` of them, in plan order, into
/// `stage` (slot k at stage + k * stride) with coalesced 16-byte loads, then points their plan entries at the
/// copies, so the unchanged expert kernels read them from VRAM.  The bytes are the same, so the results are the same.
/// The groups past `cap` keep their mirror address.  Two launches on `stream`, the plan's group count read on the
/// device (`n_groups`), so it can be captured in a graph.  `wgs_per_blob`: work-groups of 256 per staged blob.
void mirror_stage_blobs(unsigned long long* ptr, const int32_t* n_groups, uint64_t arena_lo, uint64_t arena_hi,
                        uint8_t* stage, int64_t stride, int64_t blob_bytes, int cap, int wgs_per_blob, void* stream);
}
