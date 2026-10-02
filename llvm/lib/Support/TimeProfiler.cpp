//===-- TimeProfiler.cpp - Hierarchical Time Profiler ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hierarchical time profiler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeProfiler.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace llvm;

namespace {

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::time_point_cast;

struct TimeTraceProfilerInstances {
  std::mutex Lock;
  std::vector<TimeTraceProfiler *> List;
};

TimeTraceProfilerInstances &getTimeTraceProfilerInstances() {
  static TimeTraceProfilerInstances Instances;
  return Instances;
}

static const char TimeTraceFileExtension[] = ".time-trace.json";

} // anonymous namespace

// Per Thread instance
static LLVM_THREAD_LOCAL TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;

TimeTraceProfiler *llvm::getTimeTraceProfilerInstance() {
  return TimeTraceProfilerInstance;
}

namespace {

using ClockType = steady_clock;
using TimePointType = time_point<ClockType>;
using DurationType = duration<ClockType::rep, ClockType::period>;
using CountAndDurationType = std::pair<size_t, DurationType>;
using NameAndCountAndDurationType =
    std::pair<std::string, CountAndDurationType>;
using NameMapEntry = StringMapEntry<CountAndDurationType>;

struct DeferredMetadata {
  TimeTraceMetadata Metadata;
  llvm::unique_function<std::string()> DetailCallback;
  llvm::unique_function<TimeTraceMetadata()> MetadataCallback;

  void resolve() {
    if (DetailCallback) {
      Metadata.Detail = DetailCallback();
      DetailCallback = nullptr;
    } else if (MetadataCallback) {
      Metadata = MetadataCallback();
      MetadataCallback = nullptr;
    }
  }
};

} // anonymous namespace

/// Represents an open or completed time section entry to be captured.
struct llvm::TimeTraceProfilerEntry {
  TimePointType Start;
  DurationType Duration{};
  NameMapEntry *NameEntry = nullptr;
  StringRef Detail;
  uint32_t MetadataIdx = UINT32_MAX;

  TimeTraceEventType EventType = TimeTraceEventType::CompleteEvent;
  uint32_t InstantEventCount = 0;
  int32_t LastChildIdx = -1;
  int32_t PrevSiblingIdx = -1;
  ClockType::rep StartUs = 0;
  ClockType::rep DurUs = 0;

  TimeTraceProfilerEntry() = default;
  TimeTraceProfilerEntry(TimePointType S, NameMapEntry *NE, StringRef Dt,
                         uint32_t MdIdx, TimeTraceEventType Et)
      : Start(S), NameEntry(NE), Detail(Dt), MetadataIdx(MdIdx), EventType(Et) {
  }

  StringRef getName() const { return NameEntry ? NameEntry->getKey() : ""; }

  // Calculate timings for FlameGraph. Strictly round down durations and
  // relative start times so sub-microsecond remainder time is attributed to
  // the parent's self-time without rounding bias.
  ClockType::rep getFlameGraphStartUs(TimePointType StartTime) const {
    return duration_cast<microseconds>(Start - StartTime).count();
  }

  ClockType::rep getFlameGraphDurUs() const {
    return duration_cast<microseconds>(Duration).count();
  }
};

// Represents a currently open (in-progress) time trace entry. InstantEvents
// that happen during an open event are associated with this parent event and
// are dropped if this event's duration is shorter than the granularity.
struct InProgressInstantEvent {
  TimePointType Time;
  NameMapEntry *NameEntry = nullptr;
  llvm::unique_function<std::string()> DetailCallback;
};

struct InProgressEntry {
  TimeTraceProfilerEntry Event;
  SmallVector<InProgressInstantEvent, 0> InstantEvents;
  SmallString<64> DetailBuffer;
  llvm::unique_function<std::string()> DetailCallback;
  llvm::unique_function<TimeTraceMetadata()> MetadataCallback;
  int32_t LastChildIdx = -1;
};

struct llvm::TimeTraceProfiler {
  TimeTraceProfiler(unsigned TimeTraceGranularity = 0, StringRef ProcName = "",
                    bool TimeTraceVerbose = false)
      : BeginningOfTime(system_clock::now()), StartTime(ClockType::now()),
        ProcName(ProcName), Pid(sys::Process::getProcessId()),
        Tid(llvm::get_threadid()), TimeTraceGranularity(TimeTraceGranularity),
        TimeTraceVerbose(TimeTraceVerbose) {
    llvm::get_thread_name(ThreadName);
  }

  std::unique_ptr<InProgressEntry> allocInProgress() {
    if (!FreeList.empty()) {
      auto Ptr = FreeList.pop_back_val();
      Ptr->InstantEvents.clear();
      Ptr->DetailBuffer.clear();
      Ptr->DetailCallback = nullptr;
      Ptr->MetadataCallback = nullptr;
      Ptr->LastChildIdx = -1;
      return Ptr;
    }
    return std::make_unique<InProgressEntry>();
  }

  TimeTraceProfilerEntry *
  begin(StringRef Name, StringRef Detail,
        TimeTraceEventType EventType = TimeTraceEventType::CompleteEvent) {
    assert(EventType != TimeTraceEventType::InstantEvent &&
           "Instant Events don't have begin and end.");
    NameMapEntry *NE = &*CountAndTotalPerName.try_emplace(Name).first;
    auto Entry = allocInProgress();
    Entry->DetailBuffer.assign(Detail);
    Entry->Event = TimeTraceProfilerEntry(TimePointType(), NE, StringRef(),
                                          UINT32_MAX, EventType);
    Stack.push_back(std::move(Entry));
    auto *Result = &Stack.back()->Event;
    Result->Start = ClockType::now();
    return Result;
  }

  TimeTraceProfilerEntry *
  begin(StringRef Name, llvm::unique_function<std::string()> Detail,
        TimeTraceEventType EventType = TimeTraceEventType::CompleteEvent) {
    assert(EventType != TimeTraceEventType::InstantEvent &&
           "Instant Events don't have begin and end.");
    NameMapEntry *NE = &*CountAndTotalPerName.try_emplace(Name).first;
    auto Entry = allocInProgress();
    Entry->DetailCallback = std::move(Detail);
    Entry->Event = TimeTraceProfilerEntry(TimePointType(), NE, StringRef(),
                                          UINT32_MAX, EventType);
    Stack.push_back(std::move(Entry));
    auto *Result = &Stack.back()->Event;
    Result->Start = ClockType::now();
    return Result;
  }

  TimeTraceProfilerEntry *
  begin(StringRef Name, llvm::unique_function<TimeTraceMetadata()> Metadata,
        TimeTraceEventType EventType = TimeTraceEventType::CompleteEvent) {
    assert(EventType != TimeTraceEventType::InstantEvent &&
           "Instant Events don't have begin and end.");
    NameMapEntry *NE = &*CountAndTotalPerName.try_emplace(Name).first;
    auto Entry = allocInProgress();
    Entry->MetadataCallback = std::move(Metadata);
    Entry->Event = TimeTraceProfilerEntry(TimePointType(), NE, StringRef(),
                                          UINT32_MAX, EventType);
    Stack.push_back(std::move(Entry));
    auto *Result = &Stack.back()->Event;
    Result->Start = ClockType::now();
    return Result;
  }

  void insert(StringRef Name, llvm::unique_function<std::string()> Detail) {
    if (Stack.empty())
      return;

    TimePointType Now = ClockType::now();
    NameMapEntry *NE = &*CountAndTotalPerName.try_emplace(Name).first;
    Stack.back()->InstantEvents.push_back({Now, NE, std::move(Detail)});
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    end(Stack.back()->Event);
  }

  void end(TimeTraceProfilerEntry &E) {
    assert(!Stack.empty() && "Must call begin() first");
    TimePointType End = ClockType::now();
    DurationType Duration = End - E.Start;
    E.Duration = Duration;

    auto *Iter =
        llvm::find_if(Stack, [&](const std::unique_ptr<InProgressEntry> &Val) {
          return &Val->Event == &E;
        });
    assert(Iter != Stack.end() && "Event not in the Stack");
    InProgressEntry &InProg = **Iter;

    // Only include sections longer or equal to TimeTraceGranularity usec.
    if (duration_cast<microseconds>(Duration).count() >= TimeTraceGranularity) {
      if (!InProg.DetailBuffer.empty()) {
        E.Detail = InternedStrings.insert(InProg.DetailBuffer).first->getKey();
      } else if (InProg.DetailCallback) {
        E.MetadataIdx = MetadataEntries.size();
        DeferredMetadata DM;
        DM.DetailCallback = std::move(InProg.DetailCallback);
        MetadataEntries.push_back(std::move(DM));
      } else if (InProg.MetadataCallback) {
        E.MetadataIdx = MetadataEntries.size();
        DeferredMetadata DM;
        DM.MetadataCallback = std::move(InProg.MetadataCallback);
        MetadataEntries.push_back(std::move(DM));
      }

      int32_t &ParentLastChild = (Iter != Stack.begin())
                                     ? (*std::prev(Iter))->LastChildIdx
                                     : LastRootChildIdx;
      int32_t Idx = Entries.size();
      E.LastChildIdx = InProg.LastChildIdx;
      E.PrevSiblingIdx = ParentLastChild;
      ParentLastChild = Idx;
      E.InstantEventCount = InProg.InstantEvents.size();
      Entries.push_back(E);
      for (auto &IE : InProg.InstantEvents) {
        uint32_t MdIdx = UINT32_MAX;
        if (IE.DetailCallback) {
          MdIdx = MetadataEntries.size();
          DeferredMetadata DM;
          DM.DetailCallback = std::move(IE.DetailCallback);
          MetadataEntries.push_back(std::move(DM));
        }
        Entries.emplace_back(IE.Time, IE.NameEntry, StringRef(), MdIdx,
                             TimeTraceEventType::InstantEvent);
      }
    }

    // Track total time taken by each "name", but only the topmost levels of
    // them; e.g. if there's a template instantiation that instantiates other
    // templates from within, we only want to add the topmost one. "topmost"
    // happens to be the ones that don't have any currently open entries above
    // itself.
    if (llvm::none_of(llvm::drop_begin(llvm::reverse(Stack)),
                      [&](const std::unique_ptr<InProgressEntry> &Val) {
                        return Val->Event.NameEntry == E.NameEntry;
                      })) {
      auto &CountAndTotal = E.NameEntry->second;
      CountAndTotal.first++;
      CountAndTotal.second += Duration;
    }

    FreeList.push_back(std::move(*Iter));
    Stack.erase(Iter);
  }

  void finalize() {
    for (DeferredMetadata &DM : MetadataEntries)
      DM.resolve();
  }

  StringRef getDetail(const TimeTraceProfilerEntry &E) const {
    if (E.MetadataIdx != UINT32_MAX)
      return MetadataEntries[E.MetadataIdx].Metadata.Detail;
    return E.Detail;
  }

  StringRef getFile(const TimeTraceProfilerEntry &E) const {
    if (E.MetadataIdx != UINT32_MAX)
      return MetadataEntries[E.MetadataIdx].Metadata.File;
    return "";
  }

  int getLine(const TimeTraceProfilerEntry &E) const {
    if (E.MetadataIdx != UINT32_MAX)
      return MetadataEntries[E.MetadataIdx].Metadata.Line;
    return 0;
  }

  void prepareEntriesForWrite() {
    finalize();

    // Compute floor-rounded microsecond timestamps and clamp child start
    // times top-down so sub-microsecond start offsets never cause a child
    // event to overrun its parent's floor-rounded end time.
    for (TimeTraceProfilerEntry &E : Entries) {
      E.StartUs = E.getFlameGraphStartUs(StartTime);
      E.DurUs = E.getFlameGraphDurUs();
    }
    for (size_t Idx = Entries.size(); Idx-- > 0;) {
      const auto &E = Entries[Idx];
      if (E.EventType == TimeTraceEventType::InstantEvent)
        continue;
      ClockType::rep PStart = E.StartUs;
      ClockType::rep PEnd = PStart + E.DurUs;
      ClockType::rep MaxEnd = PEnd;
      for (uint32_t I = 0; I < E.InstantEventCount; ++I) {
        auto &IE = Entries[Idx + 1 + I];
        IE.StartUs = std::clamp(IE.StartUs, PStart, PEnd);
      }
      TimePointType NextRawStart = E.Start + E.Duration;
      for (int32_t C = E.LastChildIdx; C != -1; C = Entries[C].PrevSiblingIdx) {
        auto &Child = Entries[C];
        if (Child.EventType == TimeTraceEventType::CompleteEvent ||
            Child.Start + Child.Duration <= NextRawStart) {
          Child.StartUs = std::min(Child.StartUs, MaxEnd - Child.DurUs);
          MaxEnd = Child.StartUs;
          NextRawStart = Child.Start;
        } else {
          Child.StartUs = std::min(Child.StartUs, PEnd - Child.DurUs);
        }
      }
    }
  }

  // Write events from this TimeTraceProfilerInstance and
  // ThreadTimeTraceProfilerInstances.
  void write(raw_pwrite_stream &OS) {
    // Acquire Mutex as reading ThreadTimeTraceProfilerInstances.
    auto &Instances = getTimeTraceProfilerInstances();
    std::lock_guard<std::mutex> Lock(Instances.Lock);
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling write");
    assert(llvm::all_of(Instances.List,
                        [](const auto &TTP) { return TTP->Stack.empty(); }) &&
           "All profiler sections should be ended when calling write");

    prepareEntriesForWrite();
    for (TimeTraceProfiler *TTP : Instances.List)
      TTP->prepareEntriesForWrite();

    json::OStream J(OS);
    J.objectBegin();
    J.attributeBegin("traceEvents");
    J.arrayBegin();

    // Emit all events for the main flame graph.
    auto writeEvent = [&](const TimeTraceProfiler &TTP,
                          const TimeTraceProfilerEntry &E) {
      auto StartUs = E.StartUs;
      auto DurUs = E.DurUs;
      StringRef Name = E.getName();
      StringRef Detail = TTP.getDetail(E);
      StringRef File = TTP.getFile(E);
      int Line = TTP.getLine(E);

      J.object([&] {
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(TTP.Tid));
        J.attribute("ts", StartUs);
        if (E.EventType == TimeTraceEventType::AsyncEvent) {
          J.attribute("cat", Name);
          J.attribute("ph", "b");
          J.attribute("id", 0);
        } else if (E.EventType == TimeTraceEventType::CompleteEvent) {
          J.attribute("ph", "X");
          J.attribute("dur", DurUs);
        } else { // instant event
          assert(E.EventType == TimeTraceEventType::InstantEvent &&
                 "InstantEvent expected");
          J.attribute("ph", "i");
        }
        J.attribute("name", Name);
        if (!Detail.empty() || !File.empty()) {
          J.attributeObject("args", [&] {
            if (!Detail.empty())
              J.attribute("detail", Detail);
            if (!File.empty())
              J.attribute("file", File);
            if (Line > 0)
              J.attribute("line", Line);
          });
        }
      });

      if (E.EventType == TimeTraceEventType::AsyncEvent) {
        J.object([&] {
          J.attribute("pid", Pid);
          J.attribute("tid", int64_t(TTP.Tid));
          J.attribute("ts", StartUs + DurUs);
          J.attribute("cat", Name);
          J.attribute("ph", "e");
          J.attribute("id", 0);
          J.attribute("name", Name);
        });
      }
    };
    for (const TimeTraceProfilerEntry &E : Entries)
      writeEvent(*this, E);
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const TimeTraceProfilerEntry &E : TTP->Entries)
        writeEvent(*TTP, E);

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    // Find highest used thread id.
    uint64_t MaxTid = this->Tid;
    for (const TimeTraceProfiler *TTP : Instances.List)
      MaxTid = std::max(MaxTid, TTP->Tid);

    // Combine all CountAndTotalPerName from threads into one.
    StringMap<CountAndDurationType> AllCountAndTotalPerName;
    auto combineStat = [&](const auto &Stat) {
      auto Value = Stat.getValue();
      if (Value.first == 0)
        return;
      StringRef Key = Stat.getKey();
      auto &CountAndTotal = AllCountAndTotalPerName[Key];
      CountAndTotal.first += Value.first;
      CountAndTotal.second += Value.second;
    };
    for (const auto &Stat : CountAndTotalPerName)
      combineStat(Stat);
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const auto &Stat : TTP->CountAndTotalPerName)
        combineStat(Stat);

    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(AllCountAndTotalPerName.size());
    for (const auto &Total : AllCountAndTotalPerName)
      SortedTotals.emplace_back(std::string(Total.getKey()), Total.getValue());

    llvm::sort(SortedTotals, [](const NameAndCountAndDurationType &A,
                                const NameAndCountAndDurationType &B) {
      return A.second.second > B.second.second;
    });

    // Report totals on separate threads of tracing file.
    uint64_t TotalTid = MaxTid + 1;
    for (const NameAndCountAndDurationType &Total : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(Total.second.second).count();
      auto Count = AllCountAndTotalPerName[Total.first].first;

      J.object([&] {
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(TotalTid));
        J.attribute("ph", "X");
        J.attribute("ts", 0);
        J.attribute("dur", DurUs);
        J.attribute("name", "Total " + Total.first);
        J.attributeObject("args", [&] {
          J.attribute("count", int64_t(Count));
          J.attribute("avg ms", int64_t(DurUs / Count / 1000));
        });
      });

      ++TotalTid;
    }

    auto writeMetadataEvent = [&](const char *Name, uint64_t Tid,
                                  StringRef arg) {
      J.object([&] {
        J.attribute("cat", "");
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(Tid));
        J.attribute("ts", 0);
        J.attribute("ph", "M");
        J.attribute("name", Name);
        J.attributeObject("args", [&] { J.attribute("name", arg); });
      });
    };

    writeMetadataEvent("process_name", Tid, ProcName);
    writeMetadataEvent("thread_name", Tid, ThreadName);
    for (const TimeTraceProfiler *TTP : Instances.List)
      writeMetadataEvent("thread_name", TTP->Tid, TTP->ThreadName);

    J.arrayEnd();
    J.attributeEnd();

    // Emit the absolute time when this TimeProfiler started.
    // This can be used to combine the profiling data from
    // multiple processes and preserve actual time intervals.
    J.attribute("beginningOfTime",
                time_point_cast<microseconds>(BeginningOfTime)
                    .time_since_epoch()
                    .count());

    J.objectEnd();
  }

  SmallVector<std::unique_ptr<InProgressEntry>, 16> Stack;
  SmallVector<std::unique_ptr<InProgressEntry>, 16> FreeList;
  SmallVector<TimeTraceProfilerEntry, 128> Entries;
  SmallVector<DeferredMetadata, 0> MetadataEntries;
  StringMap<CountAndDurationType> CountAndTotalPerName;
  StringSet<BumpPtrAllocator> InternedStrings;
  int32_t LastRootChildIdx = -1;
  // System clock time when the session was begun.
  const time_point<system_clock> BeginningOfTime;
  // Profiling clock time when the session was begun.
  const TimePointType StartTime;
  const std::string ProcName;
  const sys::Process::Pid Pid;
  SmallString<0> ThreadName;
  const uint64_t Tid;

  // Minimum time granularity (in microseconds)
  const unsigned TimeTraceGranularity;

  // Make time trace capture verbose event details (e.g. source filenames). This
  // can increase the size of the output by 2-3 times.
  const bool TimeTraceVerbose;
};

bool llvm::isTimeTraceVerbose() {
  return getTimeTraceProfilerInstance() &&
         getTimeTraceProfilerInstance()->TimeTraceVerbose;
}

void llvm::timeTraceProfilerInitialize(unsigned TimeTraceGranularity,
                                       StringRef ProcName,
                                       bool TimeTraceVerbose) {
  assert(TimeTraceProfilerInstance == nullptr &&
         "Profiler should not be initialized");
  TimeTraceProfilerInstance = new TimeTraceProfiler(
      TimeTraceGranularity, llvm::sys::path::filename(ProcName),
      TimeTraceVerbose);
}

// Removes all TimeTraceProfilerInstances.
// Called from main thread.
void llvm::timeTraceProfilerCleanup() {
  delete TimeTraceProfilerInstance;
  TimeTraceProfilerInstance = nullptr;

  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  for (auto *TTP : Instances.List)
    delete TTP;
  Instances.List.clear();
}

// Finish TimeTraceProfilerInstance on a worker thread.
// This doesn't remove the instance, just moves the pointer to global vector.
void llvm::timeTraceProfilerFinishThread() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->finalize();
  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  Instances.List.push_back(TimeTraceProfilerInstance);
  TimeTraceProfilerInstance = nullptr;
}

void llvm::timeTraceProfilerFinalize() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->finalize();

  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  for (auto *TTP : Instances.List)
    TTP->finalize();
}

void llvm::timeTraceProfilerWrite(raw_pwrite_stream &OS) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");
  TimeTraceProfilerInstance->write(OS);
}

Error llvm::timeTraceProfilerWrite(StringRef PreferredFileName,
                                   StringRef FallbackFileName) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");

  std::string Path = PreferredFileName.str();
  if (Path.empty()) {
    Path = FallbackFileName == "-" ? "out" : FallbackFileName.str();
    Path += TimeTraceFileExtension;
  }

  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_TextWithCRLF);
  if (EC)
    return createStringError(EC, "Could not open " + Path);

  timeTraceProfilerWrite(OS);
  return Error::success();
}

TimeTraceProfilerEntry *llvm::timeTraceProfilerBegin(StringRef Name,
                                                     StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(Name, Detail,
                                            TimeTraceEventType::CompleteEvent);
  return nullptr;
}

TimeTraceProfilerEntry *
llvm::timeTraceProfilerBegin(StringRef Name,
                             llvm::unique_function<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(Name, std::move(Detail),
                                            TimeTraceEventType::CompleteEvent);
  return nullptr;
}

TimeTraceProfilerEntry *llvm::timeTraceProfilerBegin(
    StringRef Name, llvm::unique_function<TimeTraceMetadata()> Metadata) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(Name, std::move(Metadata),
                                            TimeTraceEventType::CompleteEvent);
  return nullptr;
}

TimeTraceProfilerEntry *llvm::timeTraceAsyncProfilerBegin(StringRef Name,
                                                          StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(Name, Detail,
                                            TimeTraceEventType::AsyncEvent);
  return nullptr;
}

void llvm::timeTraceAddInstantEvent(
    StringRef Name, llvm::unique_function<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->insert(Name, std::move(Detail));
}

void llvm::timeTraceProfilerEnd() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end();
}

void llvm::timeTraceProfilerEnd(TimeTraceProfilerEntry *E) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end(*E);
}
