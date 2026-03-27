import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
import warnings
import logging
import sys
import config
import signal
from src.utils.logger import logger
# 抑制所有警告
warnings.filterwarnings("ignore")

# 设置pdfplumber的日志级别为ERROR，避免FontBBox警告
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# 导入pdfplumber
import pdfplumber

class PDFBatchProcessor:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    def find_pdf_files(self, input_path: str) -> List[Path]:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            return [path]
        elif path.is_dir():
            pdf_files = list(path.glob("**/*.pdf"))
            logger.info(f"在 {input_path} 中找到 {len(pdf_files)} 个PDF⽂件")
            return pdf_files
        else:
            raise ValueError(f"路径不存在,或不是PDF⽂件: {input_path}")

    def extract_pdf_content(self,
                           pdf_path: Path,
                           extract_text: bool = True,
                           extract_tables: bool = True,
                           table_settings: Optional[dict] = None,
                           page_timeout: int = 30) -> Dict:
        """提取PDF内容，为每个页面设置超时避免卡住"""
        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "metadata": {},
            "pages": [],
            "error": None
        }
        
        def timeout_handler(signum, frame):
            raise TimeoutError("页面处理超时")
        
        # 重定向stderr来捕获警告
        import io
        from contextlib import redirect_stderr
        
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stderr(stderr_capture):
                with pdfplumber.open(pdf_path) as pdf:
                    result["metadata"] = pdf.metadata
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_result = {"page_number": page_num, "text": "", "tables": []}
                        
                        # 设置超时处理
                        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(page_timeout)
                        
                        try:
                            if extract_text:
                                try:
                                    text = page.extract_text(layout=False,
                                                             x_tolerance=2,
                                                             y_tolerance=2)
                                    page_result["text"] = text if text else ""
                                except Exception as e:
                                    logger.info(f"警告: 页面 {page_num} 文本提取失败: {str(e)}")
                                    page_result["text"] = ""
                            
                            if extract_tables:
                                try:
                                    tables = page.extract_tables(table_settings or {})
                                    if tables:
                                        page_result["tables"] = tables
                                except Exception as e:
                                    logger.info(f"警告: 页面 {page_num} 表格提取失败: {str(e)}")
                                    page_result["tables"] = []
                        
                        except TimeoutError:
                            logger.info(f"警告: 页面 {page_num} 处理超时，跳过此页面")
                            page_result["text"] = ""
                            page_result["tables"] = []
                        
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, original_handler)
                        
                        result["pages"].append(page_result)
        
        except TimeoutError:
            error_msg = f"处理文件超时: {pdf_path}"
            result["error"] = error_msg
            logger.info(f"错误: {error_msg}")
        except Exception as e:
            error_msg = f"处理文件失败 {pdf_path}: {str(e)}"
            result["error"] = error_msg
            logger.info(f"错误: {error_msg}")
        
        return result
    def process_batch(self, pdf_files: List[Path],
                     save_format: str = "excel",
                     **extract_kwargs) -> pd.DataFrame:
        all_results = []
        
        for i, pdf_file in enumerate(tqdm(pdf_files, desc="处理PDF文件"), 1):
            logger.info(f"处理进度: {i}/{len(pdf_files)} - {pdf_file.name}")
            try:
                # 为每个文件设置更长的超时时间
                result = self.extract_pdf_content(pdf_file, page_timeout=60, **extract_kwargs)
                all_results.append(result)
                
                if i % 10 == 0:
                    self._save_intermediate_results(all_results, f"batch_{i}")
            except KeyboardInterrupt:
                logger.info(f"警告: 处理被用户中断: {pdf_file.name}")
                raise
            except Exception as e:
                logger.info(f"错误: 处理文件 {pdf_file.name} 时出错: {str(e)}")
                all_results.append({
                    "file_name": pdf_file.name,
                    "file_path": str(pdf_file),
                    "error": f"处理失败: {str(e)}",
                    "pages": []
                })
        
        return self._save_results(all_results, save_format)

    def _save_results(self, results: List[Dict], format: str) -> pd.DataFrame:
        flat_data = []
        
        for result in results:
            if result["error"]:
                flat_data.append({
                    "file_name": result["file_name"],
                    "status": "Error",
                    "error_message": result["error"],
                    "page_count": 0,
                    "text_length": 0,
                    "table_count": 0
                })
                continue
            
            total_text = ""
            total_tables = 0
            for page in result["pages"]:
                total_text += page["text"]
                total_tables += len(page["tables"])
            
            flat_data.append({
                "file_name": result["file_name"],
                "status": "Success",
                "error_message": "",
                "page_count": len(result["pages"]),
                "text_length": len(total_text),
                "table_count": total_tables,
                "author": result["metadata"].get("Author", ""),
                "creation_date": result["metadata"].get("CreationDate", "")
            })
        
        df = pd.DataFrame(flat_data)
        
        if format.lower() == "excel":
            df.to_excel(self.output_dir / "pdf_extraction_summary.xlsx", index=False)
            
            detailed_results = []
            for result in results:
                if not result["error"]:
                    for page in result["pages"]:
                        if page["text"]:
                            detailed_results.append({
                                "file_name": result["file_name"],
                                "page_number": page["page_number"],
                                "text_content": page["text"]
                            })
            
            if detailed_results:
                pd.DataFrame(detailed_results).to_excel(
                    self.output_dir / "pdf_detailed_text.xlsx", index=False
                )
        
        elif format.lower() == "csv":
            df.to_csv(self.output_dir / "pdf_extraction_summary.csv", index=False)
        
        logger.info(f"结果已保存到 {self.output_dir}")
        return df

    def _save_intermediate_results(self, results: List[Dict], batch_name: str):
        try:
            temp_df = pd.DataFrame([{
                "file_name": r["file_name"],
                "status": "Error" if r["error"] else "Success",
                "pages_processed": len(r["pages"])
            } for r in results])
            temp_df.to_csv(self.output_dir / f"progress_{batch_name}.csv", index=False)
        except Exception as e:
            logger.info(f"警告: 保存中间结果失败: {str(e)}")

ADVANCED_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 10,
    "edge_min_length": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1
}

def main():
    # 进一步抑制所有可能的警告
    import warnings
    warnings.filterwarnings("ignore")
    
    # 设置更严格的日志过滤
    logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)
    
    processor = PDFBatchProcessor(output_dir=config.pdf_output)
    
    try:
        pdf_files = processor.find_pdf_files(config.pdf_input)
        if not pdf_files:
            logger.info("警告: 未找到PDF文件")
            return
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
        
        results_df = processor.process_batch(
            pdf_files,
            save_format="excel",
            extract_text=True,
            extract_tables=True,
            table_settings=ADVANCED_TABLE_SETTINGS
        )
        
        success_count = len(results_df[results_df["status"] == "Success"])
        logger.info(f"处理完成: {success_count}/{len(pdf_files)} 个文件成功")
        
        if success_count > 0:
            avg_text_length = results_df[results_df["status"] == "Success"]["text_length"].mean()
            avg_tables = results_df[results_df["status"] == "Success"]["table_count"].mean()
            logger.info(f"平均每文件: {avg_text_length:.0f} 字符, {avg_tables:.1f} 个表格")
        else:
            logger.info("警告: 没有文件处理成功")
    
    except KeyboardInterrupt:
        logger.info("警告: 处理被用户中断")
    except Exception as e:
        logger.info(f"错误: 处理过程发生错误: {str(e)}")

if __name__ == "__main__":
    main()
