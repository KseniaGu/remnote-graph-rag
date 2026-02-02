from pathlib import Path
from typing import Optional

import fitz
import paddle
from paddleocr import PPStructureV3

from backend.configs.constants import PDF_PAGES_NUM_MAX

# Need paddlepaddle-gpu version installed
GPU_AVAILABLE = paddle.device.is_compiled_with_cuda()


class PaddleOCRPipeline:
    """Pipeline for OCR processing of PDFs and images using PaddleOCR."""

    def __init__(self) -> None:
        """Initializes the PaddleOCR pipeline."""
        self.pipeline = PPStructureV3(device="gpu" if GPU_AVAILABLE else "cpu")

    def parse_pdf(self, input_file: Path) -> Optional[Path]:
        """Parses a PDF file and converts it to Markdown format.
        
        Args:
            input_file: Path to the input PDF file.
            
        Returns:
            Path to the generated markdown file, or None if the PDF has too many pages.
        """
        # TODO: uncomment saving images if not overkill
        mkd_file_path = input_file.with_suffix(".md")
        if mkd_file_path.exists():
            return mkd_file_path

        with fitz.open(input_file, filetype="pdf") as doc:
            pages_num = len(doc)

        if pages_num > PDF_PAGES_NUM_MAX:
            return None

        output = self.pipeline.predict(input=input_file)

        markdown_list = []
        # markdown_images = []

        for res in output:
            md_info = res.markdown
            markdown_list.append(md_info)
            # markdown_images.append(md_info.get("markdown_images", {}))

        markdown_texts = self.pipeline.concatenate_markdown_pages(markdown_list)
        # input_file_stem = Path(input_file).stem
        # mkd_file_path = output_dir / f"{input_file_stem}.md"
        # mkd_file_path = output_dir / input_file_stem / f"{input_file_stem}.md"

        with open(mkd_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_texts)

        # for item in markdown_images:
        #     if item:
        #         for path, image in item.items():
        #             file_path = output_dir / input_file_stem / path
        #             file_path.parent.mkdir(parents=True, exist_ok=True)
        #             image.save(file_path)
        return mkd_file_path

    def parse_image(self, input_file: Path) -> Path:
        """Parses an image file and converts it to Markdown format.
        
        Args:
            input_file: Path to the input image file.
            
        Returns:
            Path to the generated markdown file.
        """
        mkd_file_path = input_file.with_suffix(".md")
        if mkd_file_path.exists():
            return mkd_file_path

        output = self.pipeline.predict(input_file)

        for res in output:
            res.print()
            res.save_to_markdown(save_path=mkd_file_path)

        return mkd_file_path
