---
name: "PROJECT: SCYLLA"
description: "High-performance institutional-grade options scanner console"
colors:
  primary: "#D4AF37"
  neutral-bg: "#0B0C0E"
  neutral-surface: "#121316"
  neutral-surface2: "#1A1C20"
  neutral-border: "#222428"
  text-primary: "#E2E2E2"
  text-muted: "#7B7D82"
  signal-call: "#E2E2E2"
  signal-put: "#BF5A5A"
typography:
  display:
    fontFamily: "Instrument Serif, Playfair Display, Georgia, serif"
    fontSize: "24px"
    fontWeight: 400
    lineHeight: 1.2
  body:
    fontFamily: "Inter, Plus Jakarta Sans, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Share Tech Mono, Courier New, monospace"
    fontSize: "11px"
    fontWeight: 400
    letterSpacing: "1px"
rounded:
  sm: "2px"
  md: "4px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.neutral-bg}"
    rounded: "{rounded.sm}"
    padding: "6px 14px"
  button-primary-hover:
    backgroundColor: "#F3CD4F"
  card:
    backgroundColor: "{colors.neutral-surface}"
    rounded: "{rounded.sm}"
    padding: "16px"
---

# Design System: PROJECT: SCYLLA

## 1. Overview

**Creative North Star: "The Leviathan's Wake"**

This design system establishes a premium, institutional-grade visual interface for PROJECT: SCYLLA. It shifts the scanner away from gamer-centric "neon cyberpunk" tropes and repositions it as an elite, high-performance financial cockpit. Spatially, it favors high data density with surgical alignment, leveraging a refined contrast ratio to ensure critical volatility insights capture focus.

This system rejects flashing RGB gradients, heavy glassmorphism blurs, circular screen scanlines, and rounded bubble layouts. Visual interest is driven instead by premium typography, clean structural hierarchy, and razor-sharp structural dividers.

**Key Characteristics:**
- **Restrained Luxury**: Graphite, silver, and gold accents used with absolute moderation.
- **Surgical Density**: Maximized tabular space with narrow borders, letting data speak for itself.
- **Organic Motion**: Transitions are restricted to state changes (socket updates, hovers) and ease out exponentially.

## 2. Colors

The color palette uses an obsidian graphite foundation with warm gold accents and cold platinum text values.

### Primary
- **Muted Gold** (`#D4AF37` / `oklch(75% 0.12 85)`): Used exclusively for primary calls-to-action, active indicators, and high-priority whale anomalies.
- **Platinum Chrome** (`#E2E2E2` / `oklch(90% 0.00 0)`): Used for primary text readouts, call options flow highlights, and selected dashboard states.

### Neutral
- **Matte Graphite** (`#0B0C0E` / `oklch(9% 0.002 240)`): Root application background.
- **Dark Satin** (`#121316` / `oklch(13% 0.004 240)`): Primary widget and card backgrounds.
- **Secondary Satin** (`#1A1C20` / `oklch(17% 0.005 240)`): Alternating row containers, filters bars, and secondary panels.
- **Border Grey** (`#222428` / `oklch(22% 0.006 240)`): Structural dividers and input frames.
- **Silver Dust** (`#7B7D82` / `oklch(55% 0.005 240)`): Muted labels, secondary indicators, and table headings.

### Named Rules
**The Rarity Rule.** The primary gold accent must not exceed 10% of any given screen area. Gold represents exceptional capital movement (Whales); overusing it dilutes its semantic meaning.

**The Signal-Contrast Rule.** Do not pair green/red neon alongside gold. Calls are represented by high-contrast platinum chrome (`#E2E2E2`), and Puts are represented by deep coral red (`#BF5A5A`). Gold remains reserved for extraordinary whale alert signals.

## 3. Typography

**Display Font:** `Instrument Serif`, `Playfair Display`, `Georgia`, serif
**Body Font:** `Inter`, `Plus Jakarta Sans`, `system-ui`, sans-serif
**Label/Mono Font:** `Share Tech Mono`, `Courier New`, monospace

The typography pairs an elegant, literary serif header with a clean, highly readable geometric sans-serif for UI labels and data grids, ensuring a high-end corporate terminal voice.

### Hierarchy
- **Display** (Regular, 24px, line-height 1.2): Used for primary dashboard title and key widget headers.
- **Headline** (Medium, 16px, line-height 1.3): Used for sub-widget headers and modal titles.
- **Title** (Semi-Bold, 13px, line-height 1.4): Table headers and card group titles.
- **Body** (Regular, 13px, line-height 1.5): Standard paragraphs, tooltips, and explanation copy.
- **Label** (Mono Regular, 11px, letter-spacing 1px): Numerical values, pricing data, DTE metrics, and status readouts.

## 4. Elevation

PROJECT: SCYLLA is flat-by-default, prioritizing high-performance rendering. The system does not use ambient blurry drop shadows on standard elements. Depth is established purely through distinct tonal layers (`#0B0C0E` background vs `#121316` panels) and sharp borders.

### Shadow Vocabulary
- **Active Focus Glow** (`box-shadow: 0 0 4px rgba(212, 175, 55, 0.25)`): Subtle gold glow used only on focused inputs and active state elements.

### Named Rules
**The Flat-By-Default Rule.** All containers and widgets sit on the same horizontal plane. Drop shadows are forbidden for card separation; separation must be achieved using the border token (`#222428`) or background contrast.

## 5. Components

### Buttons
- **Shape**: Rectangular with sharp corners (2px radius).
- **Primary**: Background muted gold (`#D4AF37`), text matte graphite (`#0B0C0E`), padding `6px 14px`.
- **Hover / Focus**: Lighter gold background (`#F3CD4F`), transition `150ms ease-out-expo`.
- **Secondary/Ghost**: Border `1px solid #7B7D82`, text `#E2E2E2`, transparent background. On hover, background becomes `rgba(226, 226, 226, 0.05)`.

### Cards / Containers
- **Corner Style**: Sharp (2px radius).
- **Background**: Dark Satin (`#121316`).
- **Border**: Thin grey border (`1px solid #222428`).
- **Internal Padding**: Consistent `16px`.

### Inputs / Fields
- **Style**: Background `#1A1C20`, border `1px solid #222428`, sharp corner (2px radius).
- **Focus**: Border shifts to gold (`#D4AF37`) with a subtle active focus glow.
- **Error**: Border shifts to coral red (`#BF5A5A`).

### Navigation & Headers
- **Style**: Top navigation strip with a solid Dark Satin (`#121316`) background and bottom border (`1px solid #222428`). Uses the Display Serif typography for titles, with status pings formatted in Mono labels.

## 6. Do's and Don'ts

### Do:
- **Do** format all financial data and option metrics in monospace font.
- **Do** highlight whale volume/OI anomalies with a solid gold tag or a fine gold border overlay.
- **Do** maintain a strict 4.5:1 contrast ratio for text elements against graphite surfaces.
- **Do** align data tables symmetrically, right-aligning numbers and left-aligning tickers.

### Don't:
- **Don't** use neon cyan, fluorescent green, or bright purple gradients anywhere on the layout.
- **Don't** use decorative scanlines or animate elements on hover (such as scale or translation changes).
- **Don't** add large rounded corners to cards (keep border-radius at 2px or 4px maximum).
- **Don't** use modals for simple configuration; prefer inline filters or progressive sidebars.
