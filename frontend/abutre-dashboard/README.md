# 🦅 Abutre Dashboard - Frontend

Real-time trading dashboard for Abutre bot (Delayed Martingale Strategy)

**Status:** 🔄 Em Desenvolvimento (FASE 2 - 15% completo)

---

## 🚀 Quick Start

### Install Dependencies

```bash
cd frontend/abutre-dashboard
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
npm start
```

---

## 🏗️ Tech Stack

- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **State:** Zustand
- **Real-time:** Socket.IO Client
- **Icons:** Lucide React

---

## 📊 Features (Planned)

### Dashboard Main
- [x] Setup Next.js + TypeScript + Tailwind
- [ ] MetricsCard (Balance, ROI, Win Rate, Drawdown)
- [ ] EquityCurve (Real-time chart)
- [ ] CurrentPosition (Monitor active trade)
- [ ] MarketMonitor (Streak countdown)
- [ ] TradesTable (Recent trades)
- [ ] System alerts

### Real-time Features
- [ ] WebSocket connection to backend
- [ ] Live balance updates
- [ ] Live candle updates
- [ ] Trade notifications
- [ ] Emergency alerts

### Settings
- [ ] Configuration panel
- [ ] Delay Threshold adjustment
- [ ] Max Level adjustment
- [ ] Auto-trading toggle

---

## 🎨 Design

**Theme:** Dark mode (Slate background)
**Colors:**
- Primary: Blue (#0ea5e9)
- Success: Green (#10b981)
- Danger: Red (#ef4444)
- Warning: Orange (#f59e0b)

---

## 📁 Project Structure

```
frontend/abutre-dashboard/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard main page
│   │   └── globals.css         # Global styles
│   │
│   ├── components/
│   │   ├── MetricsCard.tsx
│   │   ├── EquityCurve.tsx
│   │   ├── CurrentPosition.tsx
│   │   ├── MarketMonitor.tsx
│   │   ├── TradesTable.tsx
│   │   └── AlertSystem.tsx
│   │
│   ├── lib/
│   │   ├── websocket-client.ts  # Socket.IO client
│   │   └── utils.ts             # Helper functions
│   │
│   ├── hooks/
│   │   ├── useDashboard.ts      # Zustand store
│   │   └── useWebSocket.ts      # WS hook
│   │
│   └── types/
│       └── index.ts             # TypeScript types
│
├── public/
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

---

## 🔌 Backend Integration

Dashboard connects to Abutre bot backend via:

**WebSocket Events:**
- `balance_update` - Balance changes
- `new_candle` - M1 candle closed
- `trigger_detected` - Streak >= 8
- `trade_opened` - New position
- `trade_closed` - Position closed
- `position_update` - Level up
- `system_alert` - Critical events

**REST API (Optional):**
- `GET /api/stats` - Current statistics
- `GET /api/trades` - Trade history
- `GET /api/equity` - Equity curve data
- `POST /api/settings` - Update config

---

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│ Header: ABUTRE DASHBOARD | Status: RUNNING | $2,805.10 │
├─────────────────────────────────────────────────────────┤
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│ │ Balance  │ │   ROI    │ │Win Rate  │ │ Max DD   │   │
│ │ $2,805   │ │ +40.25%  │ │  100%    │ │ 24.81%   │   │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│ ┌───────────────────────┐ ┌───────────────────────┐   │
│ │  Equity Curve         │ │ Current Position      │   │
│ │  (Recharts line)      │ │ Status: In Position   │   │
│ │                       │ │ Direction: SELL       │   │
│ │                       │ │ Level: 3              │   │
│ │                       │ │ Stake: $4.00          │   │
│ └───────────────────────┘ └───────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│ ┌───────────────────────┐ ┌───────────────────────┐   │
│ │  Market Monitor       │ │ Recent Trades         │   │
│ │  V100: 1234.56        │ │ #1018 WIN   +$0.95    │   │
│ │  Streak: 🟢 5 velas  │ │ #1017 WIN   +$7.60    │   │
│ │  Countdown: 3 to 8    │ │ #1016 WIN   +$0.95    │   │
│ └───────────────────────┘ └───────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Next Steps (FASE 2 Continuação)

1. Implementar componentes principais (MetricsCard, EquityCurve, etc.)
2. Criar WebSocket client (real-time connection)
3. Implementar Zustand store (state management)
4. Adicionar página de configurações
5. Testes de integração com backend

---

## 📝 Notes

- Frontend está sendo desenvolvido em paralelo com backend
- Backend deve expor WebSocket server para comunicação real-time
- Use `BACKEND_URL` e `WS_URL` em `.env.local` para configurar endpoints

**Expected Backend:**
- HTTP: `http://localhost:8000`
- WebSocket: `ws://localhost:8000`

---

**Status:** Infraestrutura Next.js completa, aguardando implementação de componentes
