# HiLEx: Hierarchical Layout Extraction for Question Papers

A Benchmark Dataset for Hierarchical Document Layout Analysis in Education

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/HiLEx-DLA/HiLEx?style=social)](https://github.com/HiLEx-DLA/HiLEx/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/HiLEx-DLA/HiLEx?style=social)](https://github.com/HiLEx-DLA/HiLEx/network/members)

## Overview 📚

**HiLEx (Hierarchical Layout Extraction)** is the first large-scale dataset designed for understanding the **layout of exam question paper images**. Traditional document layout benchmarks (e.g., invoices, forms, research papers) do not capture the unique structure of educational exam papers. HiLEx fills this gap by providing a rich, hierarchical annotation of question paper pages, enabling researchers and developers to train and evaluate models that parse complex exam document layouts. The dataset consists of nearly 2,000 pages of real exam content with detailed annotations for structural elements like questions, answer choices, instructions, and more.

HiLEx was curated from **eight major competitive exams** (including GMAT, GRE, SAT, JEE, UPSC, GATE, BANK, and UGC-NET), ensuring diversity in layouts and content. Each page is manually annotated by experts using a **six-class hierarchical schema**, capturing the nested structure of question papers (questions within sections, answer blocks within questions, etc.). An extensive quality control process led to **high annotation consistency** (Cohen's κ > 0.90), making HiLEx a reliable benchmark for document layout analysis in education. By enabling accurate layout extraction from question sheets, HiLEx opens up opportunities for automated exam digitization, intelligent grading systems, content retrieval for study aids, and other educational AI applications.

Additionally, we provide **baseline benchmarks** on HiLEx using a range of modern object detection models. These include one-stage detectors (YOLOv8 and variants), a two-stage detector (Detectron2), transformer-based models (DETR, RT-DETR), and vision-language models (Florence-2, PaLI-Gemma2). Our results give insight into the challenge of hierarchical layout detection: for example, the best model (Detectron2) achieves ~94% mAP@50, while certain fine-grained classes like instructions remain challenging. These benchmarks serve as a starting point for researchers to build upon and improve layout understanding in this domain.

**Published at ICDAR 2025** — [Springer Link](https://link.springer.com/chapter/10.1007/978-3-032-04627-7_28)

## Dataset Highlights 🔢

- **Total Pages:** 1,965 exam question paper pages (single- and multi-column layouts)
- **Exams Covered:** 8 diverse exams – GMAT, GRE, SAT, JEE, GATE, UPSC, UGC-NET, and BANK (all English language)
- **Annotated Classes:** 6 hierarchical layout categories per page (see Annotation Hierarchy below)
- **Annotations:** High-quality bounding boxes for each layout element (verified by experts, inter-annotator κ > 0.90)
- **Formats:** Annotations provided in **YOLO** format (per-image .txt files) and **COCO** format (consolidated JSON) for compatibility with different training pipelines
- **License:** Released under Creative Commons **CC BY 4.0** (free to use with attribution)

## Annotation Hierarchy 🏷

Each document image is annotated with a six-class hierarchical schema that reflects the logical structure of question papers. The classes and their meanings are:

- **Question\_Paper\_Area:** The full content region of the page (the entire question paper content on that page).
- **Question\_Answer\_Block:** A combined region containing both a question and its corresponding answer choices/options (typically one complete Q&A item).
- **Question\_Block:** The question text region only (the problem statement, without any answer options).
- **Answer\_Block:** The answer choice section for a question (e.g. multiple-choice options or designated answer area).
- **Instruction:** General instructions or guidelines, often found at the top of the paper or at section beginnings (e.g. directions to candidates, section instructions).
- **Description:** Explanatory text or solutions related to a question (e.g. an explanation of the answer, often found in solution keys or detailed answers).

These classes capture a hierarchy: for instance, a **Question\_Answer\_Block** will usually contain a **Question\_Block** and an **Answer\_Block** within it, and all content lies within the overall **Question\_Paper\_Area**. This structured labeling enables models to understand not just where elements are, but how they relate hierarchically (which is crucial for parsing complex exam layouts).

## Download & Usage ⬇

The **HiLEx dataset is publicly available** via this repository. You can obtain the data by cloning the repo or downloading the release package from the [HiLEx GitHub page](https://github.com/HiLEx-DLA/HiLEx). The dataset includes all images and annotations needed to train and evaluate layout analysis models.

**Contents of the dataset:**

- 🖼 **Images:** 1,965 scanned question paper page images (JPEG format) from the 8 exams listed above.
- 🗂 **Annotations:** Bounding box annotations for each page, covering all six layout classes.
- 📄 **Formats:** Provided in both YOLO format (text files for each image) and COCO format (a JSON file) for easy integration with popular frameworks.
- 🧾 **Documentation:** This README, class definitions, and sample visualization scripts to help get started.

Once downloaded, the repository is structured as follows (simplified view):

```plaintext
HiLEx/
├── images/               # Raw question paper images (JPEG files)
└── annotations/
    ├── yolo/             # YOLO format annotations (.txt files for each image)
    └── coco/             # COCO format annotations (e.g., a single JSON file)
```

Each image file in `images/` has a corresponding `.txt` annotation in `annotations/yolo/`. The YOLO annotation files are in the standard YOLOv5/Ultralytics format: each line contains `<class_id> <x_center> <y_center> <width> <height>` with coordinates normalized to the image dimensions. These can be directly used with YOLO-based detectors (e.g., YOLOv5/YOLOv8 training code). The COCO annotation (in the `annotations/coco/` folder) provides a single JSON with bounding boxes and category labels for all images, compatible with frameworks like Detectron2 or MMDetection.

**Using the dataset:** You can train models by pointing your training scripts to the `images/` directory and using either the YOLO annotations (if training with YOLO models) or the COCO JSON (for two-stage or transformer models). We also include a class mapping and sample code in the repository to help parse the annotations. For example, to train a YOLOv8 model on HiLEx, you can use the Ultralytics YOLOv8 training command after placing the images and labels in the appropriate folder structure. Similarly, the COCO format allows easy loading via the COCO API for evaluation or for training models like Faster R-CNN.

## Benchmark Results 📊

We evaluated HiLEx with a suite of detection models spanning four major paradigms: one-stage detectors, two-stage detectors, transformers, and vision-language models. All models were trained (or used zero-shot, where noted) on the HiLEx training set and evaluated on a held-out test set, using standard detection metrics **mAP@50** and **mAP@50–95** (mean Average Precision at IoU threshold 0.5, and averaged over 0.5 to 0.95). The table below summarizes the performance of representative models:

| **Model**   | **Type**                 | **Training** | **mAP@50** | **mAP@50–95** |
|-------------|--------------------------|--------------|----------:|-------------:|
| Detectron2  | Two-Stage (Faster R-CNN) | Fine-tuned   | **94.0%** |         67.3% |
| YOLOv8      | One-Stage                | Fine-tuned   |      84.8% |         66.1% |
| YOLOv11     | One-Stage                | Fine-tuned   |      82.2% |     **68.9%** |
| PaLI-Gemma2 | Vision-Language          | Zero-shot    |      83.5% |         59.0% |
| Florence-2  | Vision-Language          | Zero-shot    |      81.0% |         59.0% |
| RT-DETR     | Transformer              | Fine-tuned   |      60.6% |         44.8% |
| DETR        | Transformer              | Fine-tuned   |      50.2% |         35.0% |

**Key takeaways:** The best overall performance is achieved by the two-stage detector **Detectron2**, which attains ~94% mAP@50 on HiLEx. Among one-stage models, **YOLOv8** performs the best (84.8% mAP@50) with fast inference and strong generalization. Notably, **YOLOv11** showed the highest localization accuracy on the strict metric (mAP@50–95 ~68.9%). Transformer-based models like DETR struggled to learn the complex layout structure without extensive tuning, while **PaLI-Gemma2** achieved ~83.5% mAP@50 *without any fine-tuning* on HiLEx, indicating promising zero-shot capabilities on structured documents.

Certain classes proved more challenging than others: **Question\_Paper\_Area** and **Question\_Answer\_Block** are generally large, distinctive regions detected with high precision by most models. In contrast, **Instruction** and **Description** had lower recall and accuracy, as models often miss or confuse these when they appear as small text segments in various parts of the page.

## Visual Samples 🖼

To illustrate the content of HiLEx and model performance, below are a few sample visualizations from the dataset and our experiments:

![Annotated sample page](./images/example_with_annotations.png)

*Figure: An example question paper page from HiLEx with ground truth annotations. Each colored bounding box corresponds to one of the six layout classes (red = Question\_Paper\_Area, blue = Question\_Block, green = Answer\_Block, etc.), clearly delineating the hierarchical sections of the page.*

![Model detection output](./images/One_Stage_model_comparison_output.png)

*Figure: Model detection output on a HiLEx page (One-Stage Detector (YOLO) Models fine-tuned on the dataset). The model's predicted boxes closely match the ground truth, correctly identifying the question area, answer options, and instructions.*

*(Additional examples and visualizations can be found in the repository, including multi-column page layouts, failure cases, and comparison of predictions across different models.)*

## Citation 📜

If you use the HiLEx dataset or benchmarks in your research, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-032-04627-7_28,
  author    = {Aich, Utathya and Chakraborty, Shinjini and Sadhukhan, Deepan and Ghosh, Swarnendu and Saha, Tulika},
  editor    = {Yin, Xu-Cheng and Karatzas, Dimosthenis and Lopresti, Daniel},
  title     = {HiLEx: Image-Based Hierarchical Layout Extraction from Question Papers},
  booktitle = {Document Analysis and Recognition -- ICDAR 2025},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {485--505},
  isbn      = {978-3-032-04627-7}
}
```

## License 📄

The HiLEx dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

## Authors 👥

| Name | Affiliation | Email |
|------|-------------|-------|
| Utathya Aich | CNH Industrial ITC, India | [us4decaich@gmail.com](mailto:us4decaich@gmail.com) |
| Shinjini Chakraborty | Institute of Engineering & Management / University of Engineering and Management, India | [shinjinic05@gmail.com](mailto:shinjinic05@gmail.com) |
| Deepan Sadhukhan | Institute of Engineering & Management / University of Engineering and Management, India | [sadhukhandeepan@gmail.com](mailto:sadhukhandeepan@gmail.com) |
| Swarnendu Ghosh | Institute of Engineering & Management / University of Engineering and Management, India | [dr.swarnendu.ghosh@gmail.com](mailto:dr.swarnendu.ghosh@gmail.com) |
| Tulika Saha *(Corresponding Author)* | International Institute of Information Technology Bangalore, India | [sahatulika15@gmail.com](mailto:sahatulika15@gmail.com) |

## Contributing 🤝

We welcome contributions from both researchers and practitioners to enhance HiLEx and its benchmarks! There are several ways you can get involved:

- **Feedback & Issues:** If you encounter any problems with the data or have suggestions for improvement, please open a GitHub Issue or contact us at [us4decaich@gmail.com](mailto:us4decaich@gmail.com).
- **New Benchmarks:** If you evaluate a new model on HiLEx and wish to have results listed, open a Pull Request with your results (model name, type, mAP@50, mAP@50–95).
- **Dataset Extensions:** We plan to expand HiLEx with non-English exam formats and OCR-based annotations — contributions and collaborations are welcome!
