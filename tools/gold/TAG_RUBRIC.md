# Tag rubric: WIZER topic tags (`articles.ai_tag`)

Tags are **finer topic facets** that sit next to the single category. An article gets **0 to 3
tags**: every tag that names a main topic of the story, and none that only names a word in it.
Judge from the headline and the description (English or Hindi). Zero tags is a normal answer:
human-interest stories, horoscopes, obituaries and most local civic news fit none.

| Tag | Use when the story is about | Not for |
|---|---|---|
| `government` | A government's actions, policies, schemes, appointments, orders, legislation (central, state, local) | Party politics and campaigning (`elections`) |
| `elections` | Elections, polls, candidates, campaigns, voter lists, Election Commission, results, party politics aimed at elections | — |
| `financial markets` | Stock markets, indices, shares, IPOs, mutual funds, bonds, commodity and currency markets, gold/silver prices | A company's own results (`corporate`) |
| `corporate` | A specific company: results, deals, mergers, launches as business news, leadership, layoffs | — |
| `monetary policy` | RBI / central banks: interest rates, repo rate, inflation targeting, liquidity, banking regulation | — |
| `economic policy` | Fiscal policy, budget, taxes and GST, trade policy and tariffs, subsidies, GDP / growth / inflation data as policy news | — |
| `cricket` | Cricket | — |
| `sports` | Any other sport | Cricket (`cricket`) |
| `entertainment` | Films, OTT, TV, music, celebrities, awards | — |
| `public health` | Outbreaks, epidemics, disease spread, vaccination drives, food safety, pollution's health effects | — |
| `healthcare` | Hospitals, doctors, treatments, medicines, health insurance and schemes, medical research | — |
| `education` | Schools, universities, exams, results, admissions, recruitment exams | — |
| `crime` | Crimes, police, arrests, investigations, courts and trials, fraud, terrorism incidents | — |
| `technology` | Gadgets, phones, apps, internet, telecom tech, product launches, cybersecurity, EV and car tech | — |
| `artificial intelligence` | AI specifically: models, chatbots, AI policy, AI companies | Generic tech (`technology`) |
| `environment` | Weather, rain, floods, cyclones, heat, pollution, climate, wildlife, forests, natural disasters | — |
| `foreign policy` | India's or any country's diplomacy, bilateral relations, summits, treaties, trade deals between countries | — |
| `conflict` | War, armed conflict, military operations, attacks between states or armed groups, defence and armed forces | Ordinary crime (`crime`) |
| `startup` | Startups, venture funding, founders, unicorns, entrepreneurship | — |
| `science` | Space / ISRO / NASA, scientific discoveries, research, science prizes | — |

Notes:
- Tags and categories are independent. A `politics` story about a new government scheme gets
  `government`; an RBI rate decision gets `monetary policy` and often `economic policy`.
- Prefer fewer, sure tags over many weak ones. Use at most 3.

For each article, output:
- **`tags`**: a list of 0–3 tag names, spelled exactly as above.
- **`cant_tell`**: `true` only when the text is too short or garbled to decide.
