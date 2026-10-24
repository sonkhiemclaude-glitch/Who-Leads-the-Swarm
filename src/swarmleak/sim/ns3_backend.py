"""ns-3.44 backend (Sections 4.2-4.3, Table 2). Requires ns-3 Python bindings (`pip install ns3`
on Linux x86_64; see docs/mac_studio_experiment_guide.md for macOS).

UAVs: IEEE 802.11a ad hoc, constant 6 Mb/s, IP MTU 1500 B, log-distance loss, OLSR or AODV,
waypoint mobility from the given trajectories (1 s waypoints, static during warm-up),
BasicEnergySource (3.0 V, 1e5 J) + WifiRadioEnergyModel. Monitors: WiFi PHY only (no IP stack,
never transmit), fixed positions; MonitorSnifferRx logs every decoded data MPDU.
Each physical message is one UDP datagram; its first 8 payload bytes carry the message index.
"""
from __future__ import annotations

import logging

import numpy as np

from ..data.schemas import FRAME_DTYPE, PhysicalMessage
from .base import SimOutput, SimSettings

log = logging.getLogger(__name__)

_CPP = r"""
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/energy-module.h"
#include <vector>
#include <cstring>
namespace swarm {
using namespace ns3;
struct Frame { double t; uint32_t size; uint64_t tx; uint64_t rx; uint16_t seq; uint8_t retry; uint32_t mon; double rssi; };
std::vector<Frame> frames;
std::vector<double> rxTime;
std::vector<Ptr<Socket>> sockets;
std::vector<Ipv4Address> addrs;
const uint16_t PORT = 9999;

void Sniff(uint32_t mon, Ptr<const Packet> p, uint16_t f, WifiTxVector v, MpduInfo m, SignalNoiseDbm sn, uint16_t sta) {
  WifiMacHeader h; Ptr<Packet> c = p->Copy();
  if (c->PeekHeader(h) == 0) return;
  if (!h.IsData()) return;
  Frame fr; fr.t = Simulator::Now().GetSeconds(); fr.size = p->GetSize();
  fr.tx = 0; fr.rx = 0; uint8_t b[6];
  h.GetAddr2().CopyTo(b); for (int i=0;i<6;i++) fr.tx = (fr.tx<<8) | b[i];
  h.GetAddr1().CopyTo(b); for (int i=0;i<6;i++) fr.rx = (fr.rx<<8) | b[i];
  fr.seq = h.GetSequenceNumber(); fr.retry = h.IsRetry() ? 1 : 0; fr.mon = mon; fr.rssi = sn.signal;
  frames.push_back(fr);
}
void Recv(Ptr<Socket> s) {
  Ptr<Packet> p; Address from;
  while ((p = s->RecvFrom(from))) {
    if (p->GetSize() < 8) continue;
    uint8_t b[8]; p->CopyData(b, 8); uint64_t id; std::memcpy(&id, b, 8);
    if (id < rxTime.size() && rxTime[id] < 0) rxTime[id] = Simulator::Now().GetSeconds();
  }
}
void SendMsg(uint32_t src, uint32_t dst, uint32_t size, uint64_t id) {
  if (size < 8) size = 8;
  std::vector<uint8_t> buf(size, 0); std::memcpy(buf.data(), &id, 8);
  Ptr<Packet> p = Create<Packet>(buf.data(), size);
  sockets[src]->SendTo(p, 0, InetSocketAddress(addrs[dst], PORT));
}
void Setup(NodeContainer uavs, Ipv4InterfaceContainer ifs, NodeContainer mons, uint32_t nmsg) {
  frames.clear(); rxTime.assign(nmsg, -1.0); sockets.clear(); addrs.clear();
  TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
  for (uint32_t i = 0; i < uavs.GetN(); i++) {
    Ptr<Socket> s = Socket::CreateSocket(uavs.Get(i), tid);
    s->Bind(InetSocketAddress(Ipv4Address::GetAny(), PORT));
    s->SetRecvCallback(MakeCallback(&Recv));
    sockets.push_back(s); addrs.push_back(ifs.GetAddress(i));
  }
  for (uint32_t k = 0; k < mons.GetN(); k++) {
    Ptr<WifiNetDevice> d = DynamicCast<WifiNetDevice>(mons.Get(k)->GetDevice(0));
    d->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx", MakeBoundCallback(&Sniff, k));
  }
}
void Schedule(const std::vector<double>& t, const std::vector<uint32_t>& s, const std::vector<uint32_t>& d,
              const std::vector<uint32_t>& sz) {
  for (size_t i = 0; i < t.size(); i++)
    Simulator::Schedule(Seconds(t[i]), &SendMsg, s[i], d[i], sz[i], (uint64_t)i);
}
void AddWaypoints(Ptr<Node> n, const std::vector<double>& t, const std::vector<double>& x,
                  const std::vector<double>& y, const std::vector<double>& z) {
  Ptr<WaypointMobilityModel> m = n->GetObject<WaypointMobilityModel>();
  for (size_t i = 0; i < t.size(); i++) m->AddWaypoint(Waypoint(Seconds(t[i]), Vector(x[i], y[i], z[i])));
}
std::vector<double> Energy(NodeContainer uavs) {
  std::vector<double> e;
  for (uint32_t i = 0; i < uavs.GetN(); i++) {
    Ptr<energy::EnergySourceContainer> c = uavs.Get(i)->GetObject<energy::EnergySourceContainer>();
    Ptr<energy::EnergySource> src = c->Get(0);
    e.push_back(src->GetInitialEnergy() - src->GetRemainingEnergy());
  }
  return e;
}
uint64_t MacOf(Ptr<Node> n) { uint8_t b[6]; Mac48Address::ConvertFrom(n->GetDevice(0)->GetAddress()).CopyTo(b);
  uint64_t v=0; for (int i=0;i<6;i++) v=(v<<8)|b[i]; return v; }
}
"""
_loaded = False


def ns3_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("ns") is not None and importlib.util.find_spec("cppyy") is not None
    except (ImportError, ValueError):
        return False


def _import_ns3():  # pragma: no cover - exercised only when ns-3 is installed
    try:
        import cppyy
        from ns import ns
    except ImportError as e:
        raise ImportError("ns-3 Python bindings not found. Install with `pip install ns3` (Linux x86_64) or "
                          "build ns-3.44 with Python bindings; see docs/mac_studio_experiment_guide.md") from e
    global _loaded
    if not _loaded:
        cppyy.cppdef(_CPP)
        _loaded = True
    return ns, cppyy


class Ns3Backend:
    name = "ns3"
    synthetic = False

    def run(self, msgs: list[PhysicalMessage], traj: np.ndarray, settings: SimSettings) -> SimOutput:  # pragma: no cover
        ns, cppyy = _import_ns3()
        sw = cppyy.gbl.swarm
        vec_d = cppyy.gbl.std.vector["double"]
        vec_u = cppyy.gbl.std.vector["unsigned int"]
        ns.RngSeedManager.SetSeed(int(settings.seed))
        ns.RngSeedManager.SetRun(int(settings.run))
        steps, n, _ = traj.shape
        duration = steps - 1
        warm = float(settings.warmup_s)
        ns.Config.SetDefault("ns3::WifiNetDevice::Mtu", ns.UintegerValue(int(settings.mtu_b)))
        uavs = ns.NodeContainer()
        uavs.Create(n)
        mons = ns.NodeContainer()
        mons.Create(len(settings.monitors))

        wifi = ns.WifiHelper()
        wifi.SetStandard(ns.WIFI_STANDARD_80211a)
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", ns.StringValue("OfdmRate6Mbps"),
                                     "ControlMode", ns.StringValue("OfdmRate6Mbps"))
        chan = ns.YansWifiChannelHelper()
        chan.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
        chan.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent",
                                ns.DoubleValue(float(settings.loss_exponent)))
        phy = ns.YansWifiPhyHelper()
        phy.SetChannel(chan.Create())
        phy.Set("TxPowerStart", ns.DoubleValue(float(settings.tx_power_dbm)))
        phy.Set("TxPowerEnd", ns.DoubleValue(float(settings.tx_power_dbm)))
        mac = ns.WifiMacHelper()
        mac.SetType("ns3::AdhocWifiMac")
        devs = wifi.Install(phy, mac, uavs)
        wifi.Install(phy, mac, mons)

        mob = ns.MobilityHelper()
        mob.SetMobilityModel("ns3::WaypointMobilityModel")
        mob.Install(uavs)
        times = np.arange(steps, dtype=float) + warm
        positions = traj
        if warm > 0:
            times = np.r_[0.0, times]
            positions = np.concatenate([traj[:1], traj], axis=0)
        tt = vec_d(times.tolist())
        for i in range(n):
            sw.AddWaypoints(uavs.Get(i), tt, vec_d(positions[:, i, 0].tolist()), vec_d(positions[:, i, 1].tolist()),
                            vec_d(positions[:, i, 2].tolist()))
        mpos = ns.ListPositionAllocator()
        for p in settings.monitors:
            mpos.Add(ns.Vector(*[float(x) for x in p]))
        mm = ns.MobilityHelper()
        mm.SetPositionAllocator(mpos)
        mm.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mm.Install(mons)

        stack = ns.InternetStackHelper()
        rh = ns.OlsrHelper() if settings.routing == "olsr" else ns.AodvHelper()
        lst = ns.Ipv4ListRoutingHelper()
        lst.Add(ns.Ipv4StaticRoutingHelper(), 0)
        lst.Add(rh, 10)
        stack.SetRoutingHelper(lst)
        stack.Install(uavs)
        ip = ns.Ipv4AddressHelper()
        ip.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.0.0"))
        ifs = ip.Assign(devs)

        esh = ns.BasicEnergySourceHelper()
        esh.Set("BasicEnergySourceInitialEnergyJ", ns.DoubleValue(1e5))
        srcs = esh.Install(uavs)
        ns.WifiRadioEnergyModelHelper().Install(devs, srcs)

        sw.Setup(uavs, ifs, mons, len(msgs))
        sw.Schedule(vec_d([m.t + warm for m in msgs]), vec_u([m.src for m in msgs]),
                    vec_u([m.dst for m in msgs]), vec_u([m.size for m in msgs]))
        ns.Simulator.Stop(ns.Seconds(duration + warm))
        ns.Simulator.Run()

        mac2node = {int(sw.MacOf(uavs.Get(i))): i for i in range(n)}
        fr = sw.frames
        F = np.zeros(len(fr), dtype=FRAME_DTYPE)
        for k in range(len(fr)):
            f = fr[k]
            F[k] = (f.t - warm, f.size, mac2node.get(int(f.tx), -1), mac2node.get(int(f.rx), -1),
                    f.seq, f.retry, f.mon, f.rssi)
        rx = np.array([sw.rxTime[i] for i in range(len(msgs))], float)
        rx = np.where(rx < 0, np.nan, rx - warm)
        energy = np.array(list(sw.Energy(uavs)), float)
        ns.Simulator.Destroy()
        log.info("ns-3 run: %d messages, %d frames", len(msgs), len(F))
        return SimOutput(F, rx, energy, self.name, False)
