---
name: ml-course-notes
description: Classify and answer the student's doubts, exam-style questions (descriptive, MCQ, match the following) and case-study questions for the "ML System Design and LLMOps" course (Priyansh Saxena, Scaler) using the prepared lecture notes. Use whenever the user asks about a lecture, week/session (e.g. W3S1), course topic (feature stores, model serving, KV cache, quantization, MLOps, drift, RAG, agents, monitoring, recsys, LLM serving), the OA1 assessment, or asks to solve or design a new ML/LLM system design case study.
---

# ML System Design & LLMOps — course notes

All paths are absolute so this works from any directory.

- Notes folder: `/Users/ommishra/Desktop/ML/class_content/lecture_notes/`
- Original slides, handouts and notebooks: `/Users/ommishra/Desktop/ML/class_content/week1/` … `week8/` (`wXsY_*`; `course_closing_slides.pdf` is in `week8/`), OA1 paper in `/Users/ommishra/Desktop/ML/class_content/assessments/`. Only open these if the notes lack the detail you need.
- Instructor's shared course website: https://www.priyanshsaxena.com/sst-ml-system-design. Refer to it if neither the notes nor the originals have what you need.

## Purpose and exam context
This skill helps the student study alongside AI to prepare for the course's two exams, which run in this order. The whole course (W1–W8) is in scope for both.
1. **Quiz (35 marks)**: descriptive, multiple-correct MCQ, match the following, case study.
2. **End-Term (45 marks)**: descriptive, single-correct MCQ, case study.

What the student will ask for, and how to help:
- **Doubts:** explain a concept in the course's own framing and numbers.
- **Practice answers:** answer a question in its exam format. Lead with the type label and then the answer. Keep any extra explanation to a line or two after it, and only if it helps.
- **Mock questions:** write new Quiz- or End-Term-style questions from the notes, using that paper's question types and marks. Give the answer key only after the student has tried them, unless they ask for it straight away.
- **Marking:** if the student pastes their own answer, mark it against the course material and list exactly what is missing or wrong.
- **Revision:** short revision sheets, topic summaries and likely-question lists for a week or topic.

Be quick: jump to the README line anchors rather than re-reading whole week files.

## Workflow

0. **Classify the query first** using `lecture_notes/exam_guide.md`: concept doubt, descriptive, single-correct MCQ, multiple-correct MCQ, match the following, or system design case study. State the type in one line, then answer in that category's format. For example, descriptive answers are a few words to at most 2 sentences per sub-part, with formula → numbers → result for calculations. The exam guide holds the instructor's evaluation rules verbatim.
1. **Always read `lecture_notes/README.md` first.** It maps each week and session, and ~20 key topics, to `file:line`, and lists known errors in the source material.
2. Open **only** the relevant `weekN.md`, starting at the line the README gives (use offset/limit; the files are 500–1000 lines). Each session is laid out as TL;DR → Concepts → Demo notebook → Handout extras → Likely doubts → Cross-links → Glossary.
3. Answer **in the course's own framing**: its terminology, numbers, formulas, tables and rules of thumb. Cite where it comes from (e.g. "W3S1, KV cache section"). Text marked `(note: ...)` in the notes is Claude's addition, not the instructor's. Say so if you rely on it.
4. If the question goes beyond the course, answer it, but mark clearly what is from the course and what is extra.
5. If the notes contradict themselves or the source, check the original PDF (`pdftotext -layout`, or read it with pages) before answering.

## Case-study questions

For any "design X" or "how did company Y build Z" question, read `lecture_notes/case_study_template.md` and follow it:
- Type A (design a system): the instructor recommends the **10-step rubric from `w5s1_part1_slides`**. Use its step names, order and checklists exactly (Part 1 of the template), and be concise and holistic: all 10 steps covered briefly (Part 2 skeleton). Lines tagged [+ from Wk 6–8 solutions] are optional enrichment.
- Type B (industry deep-dive): compare → systems side by side → per-company pipeline → compare table → lessons.
- Use the closest worked example the template points to (week5–week8) as the style reference.

## Assessments
`lecture_notes/OA1_assessment.md` has the OA1 paper (W1–W3) with the official key and topic tags. Two arithmetic errors in the key are flagged there.

## Writing answers to files
If the user asks to save an answer, design or revision sheet, write it to `/Users/ommishra/Desktop/ML/class_content/answers/<short-kebab-topic>.md`. Create the folder if needed. Never modify the files in `lecture_notes/` unless the user asks you to update the notes.
