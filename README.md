# Churro MLX

<img align="right" src="./assets/vibecoded.png" width="200">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A fast, efficient MLX-based implementation of [Churro](https://github.com/stanford-oval/Churro), a specialized vision-language model for high-accuracy historical document OCR. This implementation uses Apple's MLX framework for optimized performance on Apple Silicon.

## About Churro

Churro is a 3B-parameter open-weight vision-language model (VLM) fine-tuned specifically for historical text recognition. It was trained on **Churro-DS**, the largest historical text recognition dataset to date, comprising 99,491 pages from 155 historical corpora spanning 22 centuries across 46 language clusters.

According to the [research paper](https://arxiv.org/abs/2509.19768), Churro achieves:

- **82.3%** normalized Levenshtein similarity on printed historical documents
- **70.1%** normalized Levenshtein similarity on handwritten historical documents
- Outperforms Gemini 2.5 Pro while being **15.5 times more cost-effective**

## Features

- 🚀 **Optimized for Apple Silicon** - Uses MLX for fast inference on Mac GPUs
- 📄 **Historical Document OCR** - Specialized for documents from the 3rd century BC to the 20th century
- 🌍 **Multilingual Support** - Handles 46 language clusters including historical variants and dead languages
- 📝 **XML Output** - Produces structured HistoricalDocument XML format
- 🎨 **Pretty Printing** - Optional syntax-highlighted output using Rich
- ⚡ **8-bit Quantization** - Efficient model size with near-lossless accuracy

## Installation

### Prerequisites

- Python 3.13 or higher
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

### Model Conversion

Before first use, you need to convert the Churro model to MLX format. This requires the original HuggingFace model:

```bash
uv run python -m mlx_vlm.convert \
  --hf-path stanford-oval/churro-3B \
  --mlx-path mlx_churro_8bit \
  -q \
  --q-bits 8
```

This will download the model from HuggingFace and convert it to an 8-bit quantized MLX format, saving it to the `mlx_churro_8bit` directory.

## Usage

### Basic Usage

Transcribe a single historical document image:

```bash
uv run python churro_cli.py --pretty --max-tokens 20000 examples/Figaro_1893.JPEG
```

### Command-Line Options

```bash
uv run python churro_cli.py [OPTIONS] IMAGE

Arguments:
  IMAGE                    Path to the historical document image

Options:
  -m, --model PATH         Path to MLX converted model directory
                           [default: mlx_churro_8bit]
  --max-tokens INTEGER     Max tokens to generate (increase for dense pages)
                           [default: 20000]
  -t, --temp FLOAT         Sampling temperature (0.0 for deterministic)
                           [default: 0.6]
  -v, --verbose            Print status messages to stderr
  -p, --pretty             Pretty print output with syntax highlighting using rich
  --help                   Show this message and exit
```

### Examples

**Basic transcription:**

```bash
uv run python churro_cli.py examples/Pascal_Response_du_provincial.JPEG
```

**With pretty-printed XML output:**

```bash
uv run python churro_cli.py examples/Figaro_1893.JPEG --pretty
```

**Verbose output with custom settings:**

```bash
uv run python churro_cli.py document.jpg \
  --max-tokens 4000 \
  --temp 0.3 \
  --verbose \
  --pretty
```

**Using a different model path:**

```bash
uv run python churro_cli.py document.jpg --model /path/to/custom/model
```

## Inference Settings

These defaults match the [official Churro inference script](https://github.com/stanford-oval/Churro/blob/main/churro_transformers_infer.py) and the model's [`generation_config.json`](https://huggingface.co/stanford-oval/churro-3B/blob/main/generation_config.json).

### Prompt

Churro uses a **single, universal prompt** for all document types. There is no per-language, per-script, or per-era variation -- the model learned to handle all of these from training data alone.

- **System message**: `"Transcribe the entiretly of this historical documents to XML format."`
- **User message**: The image only, with no additional text.

> **Note:** The typos in the system message ("entiretly", "documents") are intentional. The model was fine-tuned with this exact string, so changing it may degrade performance.

### Generation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | 0.6 | Balances accuracy with diversity; use 0.0 for deterministic output |
| `repetition_penalty` | 1.05 | Prevents "repetitive degeneration" (the model looping on the same lines) |
| `max_tokens` | 20,000 | Dense pages (newspapers, ledgers) can need 4000-6000+ tokens |

### Image Preprocessing

| Setting | Value |
|---------|-------|
| Max dimension | 2500 px (longest side, aspect ratio preserved) |
| Resize method | LANCZOS downsampling |
| Color mode | RGB |
| Processor min pixels | 512 x 28 x 28 (401,408 px) |
| Processor max pixels | 5120 x 28 x 28 (4,014,080 px) |

## Output Format

Churro outputs transcriptions in **HistoricalDocument XML**, a structured schema ([`historical_doc.xsd`](https://github.com/stanford-oval/Churro/blob/main/evaluation/historical_doc.xsd)) designed to capture complex layouts, scribal edits, and missing text while preserving reading order.

### Example Output

```xml
<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata>
    <Language>French</Language>
    <WritingDirection>ltr</WritingDirection>
    <PhysicalDescription>Single printed page from a book.</PhysicalDescription>
  </Metadata>
  <Page>
    <Header>
      <PageNumber>1</PageNumber>
      <Heading type="main">
        <Line>RESPONSE DV PROVINCIAL</Line>
      </Heading>
    </Header>
    <Body>
      <Paragraph>
        <Line>MONSIEVR,</Line>
        <Line>Vos deux lettres n'ont pas esté pour moy seul...</Line>
      </Paragraph>
    </Body>
  </Page>
</HistoricalDocument>
```

The `--pretty` flag enables syntax-highlighted XML output with line numbers for better readability.

### XML Schema Elements

**Document structure:**
`HistoricalDocument` > `Metadata` + `Page` > `Header` / `Body` / `Footer`

**Metadata:** `Language`, `Script`, `WritingDirection` (ltr / rtl / ttb-ltr / ttb-rtl), `PhysicalDescription`, `Description`, `TranscriptionNote`

**Body-level elements:**

| Element | Purpose |
|---------|---------|
| `Paragraph` | Standard text block, contains `Line` elements |
| `MarginalNote` | Margin annotations (placement: left/right/top/bottom) |
| `Table` | Tabular data (`TableRow` > `TableCell` with colspan/rowspan) |
| `Heading` | Section headings (type: main / sub / running_title / figure) |
| `DateLine` | Date lines at the start of letters or entries |
| `DatedEntry` | Diary-style dated entries |
| `RecordEntry` | Log or record book entries |
| `BlockQuotation` | Quoted text (type: prose / verse) |
| `List` > `Item` | Ordered or unordered lists |
| `Figure` | Illustrations with optional `Caption` and `Description` |
| `Formula` | Mathematical/chemical formulas (LaTeX notation) |
| `Gap` | Missing text (reason: illegible / missing / damaged / omitted) |

**Inline markup (within `Line`):**

| Element | Purpose |
|---------|---------|
| `Initial` | Decorated/drop capitals (type: simple / decorated / drop / decorated-drop) |
| `Emphasis` | Styled text (type: italic / bold / underline / colored) |
| `Illegible` | Unreadable spans (reason: faded / damaged / blot / scribbled / binding) |
| `Deletion` | Struck-out text |
| `Addition` | Scribal insertions |
| `Above` | Superscript / above-line text |

**Footer-specific:** `CatchWord`, `SignatureMark`, `FolioNumber`, `PageNumber`

## Model Details

- **Base Model**: [Qwen 2.5 VL 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), fine-tuned on 97,151 training pages
- **Quantization**: 8-bit (9.8 effective bits/weight; embeddings and norms kept at full precision)
- **Disk size**: ~4.3 GB
- **Supported scripts**: 14 scripts across 5 families -- Latin, Cyrillic, Greek, Hebrew, Arabic, Devanagari, Bengali, Khmer, Japanese, Chinese, and more

### Accuracy by Document Type

From the [research paper](https://arxiv.org/abs/2509.19768), measured as normalized Levenshtein similarity:

**Printed documents** (82.3% overall):

| Strongest | Score | Weakest | Score |
|-----------|-------|---------|-------|
| Slovenian | 97.6% | Chinese | 6.2% (only 6 training samples) |
| Bulgarian | 96.1% | Sanskrit | 36.8% |
| Czech | 95.6% | Hebrew | 59.3% |

**Handwritten documents** (70.1% overall):

| Strongest | Score | Weakest | Score |
|-----------|-------|---------|-------|
| Catalan | 90.2% | Sanskrit | 21.5% |
| Italian | 88.4% | Khmer | 25.7% |
| German | 83.1% | Hebrew | 42.3% |

### Known Failure Modes

The paper documents several failure modes to be aware of:

- **Reading order errors in vertical scripts**: East Asian documents with top-to-bottom writing may have lines transcribed in the wrong order
- **Small character recognition**: Very small text (e.g., footnotes, marginalia) can be misread
- **Repetitive degeneration**: The model can get stuck repeating lines -- mitigated by `repetition_penalty: 1.05`
- **Hallucinations from stereotypes**: Occasionally generates plausible but fabricated text for damaged/illegible sections
- **Historical script changes**: Characters that changed form over centuries (e.g., long s "ſ" vs. "s") may be inconsistently transcribed

## Performance Tips

1. **Dense pages**: The default `--max-tokens 20000` handles most documents, but you can lower it for faster inference on short pages
2. **Deterministic output**: Use `--temp 0.0` for reproducible results
3. **Large images**: Images are automatically resized to fit within 2500x2500 pixels while preserving aspect ratio
4. **GPU acceleration**: MLX automatically uses Apple Silicon GPUs when available
5. **Quantization tradeoff**: 8-bit produces near-identical output to fp16 at 60% the memory; 4-bit saves more memory but introduces noticeable transcription differences

## Limitations

- Currently optimized for Apple Silicon (M-series chips)
- Model conversion required before first use
- Best performance on documents from the 3rd century BC to the 20th century
- Underrepresented languages (especially those with minimal training data) will have lower accuracy
- No African languages are represented in the training data
- Non-Latin scripts may have higher reading-order error rates

## Citation

If you use Churro in your research, please cite the original paper:

```bibtex
@inproceedings{semnani2025churro,
  title        = {{CHURRO}: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition},
  author       = {Semnani, Sina J. and Zhang, Han and He, Xinyan and Tekg{\"u}rler, Merve and Lam, Monica S.},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
  year         = {2025}
}
```

## Related Projects

- **[Original Churro Repository](https://github.com/stanford-oval/Churro)** - PyTorch implementation with full training and evaluation code
- **[Churro-DS Dataset](https://github.com/stanford-oval/Churro)** - The historical document dataset used for training
- **[MLX VLM](https://github.com/ml-explore/mlx-examples/tree/main/vlm)** - MLX vision-language model framework

## License

- **Model Weights**: Qwen research license (see [HuggingFace model card](https://huggingface.co/stanford-oval/churro-3B))
- **Dataset**: Research purposes only (see original repository for details)
- **Code**: Apache 2.0

## Acknowledgments

This MLX implementation is based on the original Churro model developed by the Stanford OVAL lab. Special thanks to the Churro team for their groundbreaking work on historical document OCR.

---

For more information about Churro, visit the [original repository](https://github.com/stanford-oval/Churro) or read the [research paper](https://arxiv.org/abs/2509.19768).
