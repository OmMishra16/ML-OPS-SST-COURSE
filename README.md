# ML System Design & LLMOps SST course notes

Study notes, exam guide and worked answers for **ML System Design and LLMOps**, taught by Priyansh Saxena at Scaler School of Technology (8 weeks, closed Sept 2026).

Instructor's course website: https://www.priyanshsaxena.com/sst-ml-system-design

## What's here

| Path | Contents |
|---|---|
| [`lecture_notes/`](lecture_notes/) | Per-week notes (`week1.md` … `week8.md`), each laid out as TL;DR → concepts → demo notebook → handout extras → likely doubts → cross-links → glossary |
| [`lecture_notes/README.md`](lecture_notes/README.md) | Index from week/session/topic to `file:line`, plus known errors in the source material |
| [`lecture_notes/exam_guide.md`](lecture_notes/exam_guide.md) | Exam format (Quiz 35 marks, End-Term 45 marks) and how to answer each question type |
| [`lecture_notes/case_study_template.md`](lecture_notes/case_study_template.md) | The instructor's 10-step ML system design rubric, for case-study answers |
| [`lecture_notes/OA1_assessment.md`](lecture_notes/OA1_assessment.md) | Online Assessment 1 (W1–W3) with answer key and topic tags |
| [`week1/`](week1/) … [`week8/`](week8/) | Original course files for each week: slides, pre/post handouts, demo notebooks, worksheets, solution designs, case studies |
| [`assessments/`](assessments/) | OA1 question paper |
| `AGENTS.md`, `CLAUDE.md`, `.claude/skills/`, `.agents/skills/` | Instructions that let AI coding agents (Claude Code, Codex, etc.) answer course doubts from these notes |

## Course map

| Week | Topics |
|---|---|
| 1 | ML Systems 101 (notebook→production gap, training-serving skew, drift, feedback loops, hidden technical debt); the ML system design framework |
| 2 | Feature stores, batch/stream pipelines, point-in-time correctness; model serving (REST/gRPC/streaming, batching, shadow/canary/blue-green) |
| 3 | LLM serving (KV cache, PagedAttention/vLLM, quantization, continuous batching, speculative decoding, API vs self-host); MLOps (MLflow, registry, CI/CD, drift, retraining) |
| 4 | RAG and vector DBs (chunking, embeddings, ANN, hybrid search/RRF, reranking); agents and tool use (ReAct, plan-and-execute, Langfuse); the four engineering layers of agents |
| 5 | The 10-step design rubric; real-time music recommender; Spotify Discover Weekly vs Netflix |
| 6 | LLM-powered semantic/hybrid search at scale; how OpenAI and Cohere serve LLM APIs |
| 7 | End-to-end enterprise RAG; LLM monitoring, drift detection and evaluation |
| 8 | Multi-model agentic customer support; bonus cases: ride-hailing ML, coding agents at scale |

## How to use

- **Revising a topic:** find it in [`lecture_notes/README.md`](lecture_notes/README.md) and jump to the line it gives.
- **Answering a question:** classify it first using [`exam_guide.md`](lecture_notes/exam_guide.md), then answer in that question type's format.
- **Case studies:** follow [`case_study_template.md`](lecture_notes/case_study_template.md) and cover all 10 steps briefly.
- **With an AI agent:** open this folder in Claude Code (or another agent that reads `AGENTS.md`) and ask your doubt. The `ml-course-notes` skill sends it to the right notes.

Text marked `(note: ...)` in the notes is the note-writer's own addition or correction, not course material.

## Disclaimer

These are personal study notes. The course content, slides and case studies belong to the instructor and Scaler School of Technology. Refer to the course website for official material.
