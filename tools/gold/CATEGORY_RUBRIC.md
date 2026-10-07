# Category rubric: WIZER news categories

Assign every article **exactly one** category: the main subject of the story as a reader would
file it. Judge from the headline and the description (when given), in whatever language they
are written. Pick the category of the main event, not of words that merely appear.

| Category | Use for | Not for |
|---|---|---|
| `politics` | Elections, parties, politicians' statements and campaigns, governments' political moves, Parliament/Assembly politics, protests with a political goal, government appointments and policy *announcements framed politically* | Court cases against politicians (→ `crime`); an economic policy explained on its economics (→ `business`) |
| `business` | Companies, markets, stocks, IPOs, results, banking, RBI and monetary policy, inflation, GDP, trade, taxes/GST, prices (gold, fuel, vegetables), jobs/salaries data, personal finance | Crypto (→ `crypto`); a company's product launch that is mainly about the technology (→ `technology`) |
| `cricket` | Anything about cricket (matches, players, IPL, BCCI, selection) | Other sports (→ `sports`) |
| `sports` | Every sport other than cricket: football, hockey, kabaddi, chess, tennis, Olympics, athletics, wrestling | Cricket (→ `cricket`) |
| `entertainment` | Films, OTT, TV, music, celebrities, Bigg Boss, box office, awards, fashion/lifestyle celebrity news | Religion/astrology (→ `general`) |
| `technology` | Gadgets, phones, apps, AI, internet, telecom tech, space/ISRO, science discoveries, cybersecurity products | Cybercrime incidents (→ `crime`); telecom tariffs/market (→ `business`) |
| `health` | Diseases, hospitals, medicine, public health, fitness, nutrition, health schemes | — |
| `education` | Schools, universities, exams and results (board, NEET, JEE, UPSC), admissions, scholarships, recruitment *exams* | Government job vacancies without an exam angle (→ `business`) |
| `crime` | Crime, police, arrests, accidents with police cases, murders, fraud, scams, terrorism incidents, **courts and judgments**, legal proceedings | Political allegations without a case (→ `politics`) |
| `environment` | Weather, rain, heat, floods, cyclones, pollution, climate, wildlife, forests, disasters from nature | — |
| `world` | International news: other countries' events, foreign affairs, wars, diplomacy, world leaders. India's foreign relations count as `world` | Indian domestic news with a foreign keyword |
| `crypto` | Cryptocurrencies, blockchain, Bitcoin, crypto exchanges and their regulation | — |
| `general` | Nothing above fits: religion, festivals, astrology/horoscopes, human interest, local civic issues (roads, water supply), obituaries, recipes, quizzes, viral stories | Use only when no specific category fits |

Tie-breaks:
- A **road or train accident**:
  - is `crime` if there is an arrest or police case;
  - is `environment` if it is a natural disaster;
  - is otherwise `general`.
- A **court ruling** on policy is `crime` (courts/legal), unless it is about an election (→ `politics`).
- **Government welfare schemes** go to the category of their domain: farmers' payments → `business`, health schemes → `health`, school schemes → `education`.
- **Horoscopes, rashifal and panchang** → `general`.

Edge cases (settled at gold-set adjudication, 2026-10-06; apply them the same way):
- **Another country's domestic events** (its crime, disasters, politics, obituaries, civic issues) →
  `world`. Exceptions: a foreign story with a clear topic keeps the topic, so companies and markets →
  `business`, science/Nobel → `technology`, sport → `sports`/`cricket`, films/celebrities →
  `entertainment`.
- **Any court case** → `crime`, including commercial disputes and service/pay cases (unless it is about
  an election → `politics`).
- **India's trade agreements and trade commissions** with another country → `world` (foreign relations).
- **Science Nobel prizes** (physics, chemistry, medicine) → `technology`; Peace → `world`; Literature and
  Economics → `general` / `business`.
- **A cricketer's personal life** (marriage, controversies off the field) → `entertainment`.
- **Accidents other than road or train** (drowning, fires, building collapse) follow the accident rule:
  `crime` if there is a police case, otherwise `general`.
- **No dedicated category:**
  - video games → `technology`
  - defence and the armed forces → `politics` (India) or `world` (foreign)
  - art, travel and tourism → `general`
  - labour protests and strikes → `politics`

For each article, output:
- **`category`**: one of the 13 categories.
- **`confidence`**: 0.5–1.0.
- **`cant_tell`**: `true` only when the text is too short, garbled or ambiguous to decide. Still give your best category.
