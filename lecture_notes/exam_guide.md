# Exam Guide — Question Types and How to Answer Each

Source: the instructor's evaluation guide (pasted by the student, Oct 2026), recorded verbatim below, plus answer conventions taken from the OA1 paper and its marking scheme (`OA1_assessment.md`).

## Instructor's guide (verbatim)
1. **Evaluation format**
   - **End-Term Exam (45 marks):** Descriptive Questions, System Design Case Study, Single Correct MCQs.
   - **Quiz for Internship-Exempted Students (35 marks):** Descriptive Questions, Multiple Correct MCQs, Match the Following, System Design Case Study. The quiz might seem slightly lengthy, but this is intentional to maintain a comparable level of evaluation for both groups (exempted as well as regular students).
2. **Marks on the platform:** the marks shown might differ slightly from the totals above, as questions have been scaled to integer marks (keeping the distribution/ratio the same) to meet platform requirements.
3. **Descriptive questions:** full marks will be awarded initially after the test. Responses will be evaluated later and marks updated. Although questions may appear lengthy, **each subpart requires a concise answer, ranging from a few words to a maximum of two sentences.**
4. **System design case study:** you are free to structure your answers as you prefer. However, the instructor recommends **following the rubric discussed in class (`w5s1_part1_slides`)** for a concise and holistic design. → `case_study_template.md`

Scope: the whole course (W1–W8). OA1 covered only W1–W3.

**The student sits BOTH papers, back to back:** first the **Quiz (35 marks)**, then the **End-Term (45 marks)** half an hour later. So:
- Prepare **all five question formats**: descriptive, single-correct MCQ, multiple-correct MCQ, match the following, case study.
- **Two case studies on the same day.** Drill the 10-step rubric skeleton until it's automatic, and practise fitting it to different domains (recsys, search/RAG, LLM serving, agents, forecasting/marketplace).
- Quiz-specific formats (multiple-correct MCQ, match the following) need option-by-option rigour, because there's no partial credit and the papers include decoys.
- In practice sessions, mix formats from both papers and keep answers exam-length.

---

## Step 0 — classify every query first
Before answering, map the question to one category and state it in one line (e.g. "**Type: Descriptive (exam-style)** · Topic: W3S1 KV cache"). Then answer in that category's format.

| # | Category | How to recognise it | Appears in |
|---|---|---|---|
| 1 | **Concept doubt / learning** | "explain", "why", "I don't understand", "difference between" | (study mode, not an exam item) |
| 2 | **Descriptive** | scenario + sub-parts (a)(b)(c)…, "calculate", "name", "explain why" | End-Term and Quiz |
| 3 | **Single-correct MCQ** | options A–D, one answer | End-Term |
| 4 | **Multiple-correct MCQ** | "select all that apply" / more than one correct | Quiz |
| 5 | **Match the following** | Column A ↔ Column B | Quiz |
| 6 | **System design case study** | "Design a system that…" with scale/latency/cost constraints | End-Term and Quiz |

Then: find the topic in `README.md` → read that `weekN.md` section → answer in the course's terminology and numbers.

---

## Answer format per category

### 1. Concept doubt (study mode)
Explain fully and intuitively, in the course's framing. Cite the session (e.g. "W2S1, point-in-time correctness"). Add an example or numbers from the slides. End with a one-line "exam version" of the answer, so the student also sees how to write it concisely.

### 2. Descriptive (exam style)
- **Per sub-part: a few words to at most two sentences.** No essays. Label each sub-part (a), (b), …
- **Numerical sub-parts:** formula → substituted numbers → result with units (OA1 rule: a correct number without working gets at most half credit; method marks apply, so errors carried forward from an earlier sub-part aren't penalised again). Keep the working on one line.
- Use the **exact course term** the question is fishing for (e.g. "point-in-time correctness violation / future leakage", "covariate shift: P(X) changes, P(Y|X) fixed", "memory-bandwidth bound").
- "Give two problems" → give exactly two, each as a short phrase plus a reason.
- "Rank / choose" → give the order, plus one clause of justification per axis asked.

### 3. Single-correct MCQ (End-Term)
Give the answer letter first. Then one line on why it is right, and a few words on why each other option is wrong (the course likes near-miss distractors).

### 4. Multiple-correct MCQ (Quiz)
Go through **every option as True/False with a one-line reason**, then give the final set (e.g. **A, C, D**). Scoring is all-or-nothing (OA1: no partial credit, no negative marking), so justify each option individually.

### 5. Match the following (Quiz)
Give the mapping as a table (`A-item → B-item(s)`). Expect **one-to-many mappings**, **reused B items** and **unused decoys, including a deliberate near-miss** (OA1 rule). Name the decoys and say why the near-miss doesn't fit.

### 6. System design case study
Follow `case_study_template.md`: the **10-step rubric from `w5s1_part1_slides`**, in the instructor's step names and order. Be **concise and holistic**: cover all 10 steps briefly rather than a few steps in depth. Do the estimates with numbers, and justify each decision ("I chose X over Y because Z; the cost is …").

---

## Practice-mode behaviour
If the student asks for practice questions, generate them in the paper's style (scenario + sub-parts with marks, MCQs with near-miss distractors, match with decoys). Use OA1 as the style reference. Mark answers using the formats above.
