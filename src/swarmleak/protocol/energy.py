"""Airtime and incremental radio-energy estimates (Section 5.3, equations for c_tx and c_rx).

    A(l)    = n_frag * (preamble + MAC ACK) + (l + n_frag * overhead) * 8 / rate
    n_frag  = ceil((l + 28) / MTU)          (28 = IPv4 + UDP headers)
    c_tx(l) = V (I_tx - I_idle) A(l)
    c_rx(l) = V (I_rx - I_idle) A(l)

l is the UDP payload of one datagram. The simulator adds a 29-byte security overhead to every
message; the scheduler's estimates include it (audit M6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import (
    I_IDLE_A,
    I_RX_A,
    I_TX_A,
    IP_MTU_B,
    IP_UDP_HEADER_B,
    MAC_ACK_S,
    PER_FRAGMENT_OVERHEAD_B,
    PHY_RATE_BPS,
    PREAMBLE_S,
    SECURITY_OVERHEAD_B,
    SUPPLY_VOLTAGE_V,
)


@dataclass(frozen=True)
class RadioModel:
    voltage_v: float = SUPPLY_VOLTAGE_V
    i_tx_a: float = I_TX_A
    i_rx_a: float = I_RX_A
    i_idle_a: float = I_IDLE_A
    rate_bps: float = PHY_RATE_BPS
    mtu_b: int = IP_MTU_B
    overhead_b: int = PER_FRAGMENT_OVERHEAD_B
    preamble_s: float = PREAMBLE_S
    ack_s: float = MAC_ACK_S
    security_overhead_b: int = SECURITY_OVERHEAD_B
    include_security_overhead: bool = True

    def fragments(self, payload_b: int) -> int:
        """Number of IP fragments for a UDP payload (Section 5.3)."""
        if payload_b <= 0:
            raise ValueError(f"payload must be positive, got {payload_b}")
        return max(1, math.ceil((payload_b + IP_UDP_HEADER_B) / self.mtu_b))

    def airtime(self, payload_b: int) -> float:
        """A(l): airtime in seconds of an l-byte datagram, all fragments (Section 5.3)."""
        nf = self.fragments(payload_b)
        return nf * (self.preamble_s + self.ack_s) + (payload_b + nf * self.overhead_b) * 8 / self.rate_bps

    def on_air_payload(self, message_b: int) -> int:
        return int(message_b) + (self.security_overhead_b if self.include_security_overhead else 0)

    def tx_cost(self, message_b: int) -> float:
        """c_tx(l) = V (I_tx - I_idle) A(l), joules (Section 5.3)."""
        return self.voltage_v * (self.i_tx_a - self.i_idle_a) * self.airtime(self.on_air_payload(message_b))

    def rx_cost(self, message_b: int) -> float:
        """c_rx(l) = V (I_rx - I_idle) A(l), joules (Section 5.3)."""
        return self.voltage_v * (self.i_rx_a - self.i_idle_a) * self.airtime(self.on_air_payload(message_b))


DEFAULT_RADIO = RadioModel()


def frame_airtime(payload_b: int) -> float:
    return DEFAULT_RADIO.airtime(payload_b)


def tx_cost(message_b: int) -> float:
    return DEFAULT_RADIO.tx_cost(message_b)


def rx_cost(message_b: int) -> float:
    return DEFAULT_RADIO.rx_cost(message_b)
