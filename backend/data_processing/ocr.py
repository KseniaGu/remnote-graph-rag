from pathlib import Path

import pymupdf as fitz
import paddle
from paddleocr import PPStructureV3

from backend.configs.constants import PDF_PAGES_NUM_MAX


# Need paddlepaddle-gpu version installed
GPU_AVAILABLE = paddle.device.is_compiled_with_cuda()
_PADDLEX_MODEL_CACHE = Path.home() / ".paddlex" / "official_models"
_TEXT_RECOGNITION_MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"
_MODEL_DIR_OVERRIDES = {
    "layout_detection_model_dir": "PP-DocLayout_plus-L",
    "region_detection_model_dir": "PP-DocBlockLayout",
    "text_detection_model_dir": "PP-OCRv5_server_det",
    "textline_orientation_model_dir": "PP-LCNet_x1_0_textline_ori",
    "text_recognition_model_dir": "eslav_PP-OCRv5_mobile_rec",
    "table_classification_model_dir": "PP-LCNet_x1_0_table_cls",
    "wired_table_structure_recognition_model_dir": "SLANeXt_wired",
    "wireless_table_structure_recognition_model_dir": "SLANet_plus",
    "wired_table_cells_detection_model_dir": "RT-DETR-L_wired_table_cell_det",
    "wireless_table_cells_detection_model_dir": "RT-DETR-L_wireless_table_cell_det",
    "table_orientation_classify_model_dir": "PP-LCNet_x1_0_doc_ori",
    "formula_recognition_model_dir": "PP-FormulaNet_plus-L",
}


def _cached_paddlex_model_dirs() -> dict[str, str]:
    model_dirs = {}
    for option_name, model_name in _MODEL_DIR_OVERRIDES.items():
        model_dir = _PADDLEX_MODEL_CACHE / model_name
        if model_dir.exists():
            model_dirs[option_name] = str(model_dir)
    return model_dirs


class PaddleOCRPipeline:
    """Pipeline for OCR processing of PDFs and images using PaddleOCR."""

    def __init__(self) -> None:
        """Initializes the PaddleOCR pipeline."""
        self.pipeline = PPStructureV3(
            device="gpu" if GPU_AVAILABLE else "cpu",
            text_recognition_model_name=_TEXT_RECOGNITION_MODEL_NAME,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
            **_cached_paddlex_model_dirs(),
        )

    def parse_pdf(self, input_file: str | Path) -> Path | None:
        """Parses a PDF file and converts it to Markdown format.
        
        Args:
            input_file: Path to the input PDF file.
            
        Returns:
            Path to the generated markdown file, or None if the PDF has too many pages.
        """
        input_file = Path(input_file)
        mkd_file_path = input_file.with_suffix(".md")
        if mkd_file_path.exists():
            return mkd_file_path

        with fitz.open(input_file, filetype="pdf") as doc:
            pages_num = len(doc)

        if pages_num > PDF_PAGES_NUM_MAX:
            return None

        output = self.pipeline.predict(input=str(input_file))

        markdown_list = []

        for res in output:
            markdown_list.append(res.markdown)

        markdown_texts = self.pipeline.concatenate_markdown_pages(markdown_list)
        with mkd_file_path.open("w", encoding="utf-8") as f:
            f.write(markdown_texts)

        return mkd_file_path

    def parse_image(self, input_file: str | Path) -> Path:
        """Parses an image file and converts it to Markdown format.
        
        Args:
            input_file: Path to the input image file.
            
        Returns:
            Path to the generated markdown file.
        """
        input_file = Path(input_file)
        mkd_file_path = input_file.with_suffix(".md")
        if mkd_file_path.exists():
            return mkd_file_path

        output = self.pipeline.predict(str(input_file))

        for res in output:
            res.save_to_markdown(save_path=str(mkd_file_path))

        return mkd_file_path
