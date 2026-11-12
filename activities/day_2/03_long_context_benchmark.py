"""Long-context recall benchmark.

Packs a folder of papers into one long context, asks the model each question
against that context, and uses a second "judge" model to grade every answer
as correct or incorrect. Results are summarized per question category.

Example:
    uv run python 03_long_context_benchmark.py \
        --papers sample_papers --questions questions.json --num-papers 10 --position middle
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

ANSWER_PROMPT = """
You are a research assistant. You will be given the full text of several
academic papers, followed by a question about one of them.
Answer the question using only information from the papers.
If the papers do not contain the answer, say that you do not know.
"""

JUDGE_PROMPT = """
You are grading answers to questions about academic papers.
You will be given a question, a reference answer, and a candidate answer.
The candidate is CORRECT if it contains the key facts of the reference answer
and does not contradict it. Extra detail is fine; missing or wrong key facts
are not. Reply with exactly one word: CORRECT or INCORRECT.
"""


def read_paper(path: Path) -> str:
    '''Returns the text of a .pdf or .txt paper.'''
    if path.suffix.lower() == '.pdf':
        return '\n'.join(page.extract_text() or '' for page in PdfReader(path).pages)
    return path.read_text()


def parse_position(value: str) -> float:
    '''Accepts start/middle/end or a number in [0, 1].'''
    named = {'start': 0.0, 'middle': 0.5, 'end': 1.0}
    try:
        position = named[value] if value in named else float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid position {value!r}')
    if not 0.0 <= position <= 1.0:
        raise argparse.ArgumentTypeError('position must be start, middle, end, or a number in [0, 1]')
    return position


def build_context(papers: dict[str, str], source: str, num_papers: int, position: float) -> tuple[str, int]:
    '''Builds a context of num_papers papers with the source paper placed at a
    relative position (0 = first, 1 = last) among them. Each paper is wrapped
    in <paper> tags so the model can tell them apart.

    The distractors are always the first num_papers - 1 non-source papers in
    sorted order, so changing position only moves the source paper.
    Returns the context and the 0-based index of the source paper.
    '''
    distractors = [name for name in sorted(papers) if name != source][:num_papers - 1]
    index = round(position * (num_papers - 1))
    order = distractors[:index] + [source] + distractors[index:]
    sections = [f'<paper file="{name}">\n{papers[name]}\n</paper>' for name in order]
    return '\n\n'.join(sections), index


def ask(client: OpenAI, model: str, context: str, question: str):
    '''Asks the model one question with the papers placed before it.
    Returns the full Response so the caller can read text and token usage.'''
    return client.responses.create(
        model=model,
        instructions=ANSWER_PROMPT,
        input=[{'role': 'user', 'content': f'{context}\n\nQuestion: {question}'}],
    )


def judge(client: OpenAI, model: str, question: str, reference: str, candidate: str) -> bool:
    '''Asks the judge model whether the candidate answer matches the reference.'''
    response = client.responses.create(
        model=model,
        instructions=JUDGE_PROMPT,
        input=[{
            'role': 'user',
            'content': (f'Question: {question}\n\nReference answer: {reference}\n\n'
                        f'Candidate answer: {candidate}'),
        }],
    )
    verdict = response.output_text.strip().upper()
    # Check INCORRECT first, since "CORRECT" is a substring of it.
    return 'INCORRECT' not in verdict and 'CORRECT' in verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--papers', type=Path, required=True, help='Folder of .pdf/.txt papers')
    parser.add_argument('--questions', type=Path, required=True, help='JSON list of {question, answer, category, source_paper}')
    parser.add_argument('-m', '--model', default='beam-1.0-bf16', help='Model under test')
    parser.add_argument('-j', '--judge-model', default='beam-1.0-bf16', help='Model used to grade answers')
    parser.add_argument('-n', '--num-papers', type=int, required=True, help='Number of papers to put in the context')
    parser.add_argument('-p', '--position', type=parse_position, default=0.5,
                        help='Where to place the source paper: start, middle, end, or 0-1 (default: middle)')
    args = parser.parse_args()

    client = OpenAI()
    questions = json.loads(args.questions.read_text())

    # Read every paper once up front; contexts are assembled per question.
    paths = sorted(p for p in args.papers.iterdir() if p.suffix.lower() in ('.pdf', '.txt'))
    if not 1 <= args.num_papers <= len(paths):
        raise SystemExit(f'--num-papers must be between 1 and {len(paths)} (papers found in {args.papers})')
    papers = {p.name: read_paper(p) for p in paths}
    for q in questions:
        if q['source_paper'] not in papers:
            raise SystemExit(f'source_paper {q["source_paper"]!r} not found in {args.papers}')

    print(f'Context: {args.num_papers} papers, source paper at position {args.position}')

    # category -> [num_correct, num_total]
    scores: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, q in enumerate(questions, 1):
        context, index = build_context(papers, q['source_paper'], args.num_papers, args.position)
        try:
            response = ask(client, args.model, context, q['question'])
            answer = response.output_text
            correct = judge(client, args.judge_model, q['question'], q['answer'], answer)
            tokens = response.usage.input_tokens if response.usage else '?'
        except Exception as e:
            # e.g. the context is longer than the model's window
            answer, correct, tokens = f'ERROR: {e}', False, '?'

        scores[q['category']][0] += correct
        scores[q['category']][1] += 1
        print(f'\n[{i}/{len(questions)}] ({q["category"]}) {"CORRECT" if correct else "INCORRECT"} '
              f'| source paper {index + 1}/{args.num_papers} | input tokens: {tokens}')
        print(f'  Q: {q["question"]}')
        print(f'  A: {answer.strip()}')

    print('\n=== Results by category ===')
    for category, (num_correct, total) in sorted(scores.items()):
        print(f'{category:>12}: {num_correct}/{total} ({100 * num_correct / total:.0f}%)')
    overall_correct = sum(c for c, _ in scores.values())
    overall_total = sum(t for _, t in scores.values())
    print(f'{"overall":>12}: {overall_correct}/{overall_total} ({100 * overall_correct / overall_total:.0f}%)')


if __name__ == '__main__':
    main()
