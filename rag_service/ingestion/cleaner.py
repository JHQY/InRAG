import re

import pandas as pd


class TableCleaner:
    """
    Clean table data extracted by pdfplumber.
    """

    def clean_cell(self, value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def clean_header(self, header):
        if header is None:
            return []

        cleaned = []
        for idx, cell in enumerate(header):
            text = self.clean_cell(cell)
            cleaned.append(text if text else f"col_{idx}")
        return cleaned

    def unify_rows(self, rows, num_cols):
        unified_rows = []
        for row in rows or []:
            values = [self.clean_cell(cell) for cell in (row or [])]
            if len(values) < num_cols:
                values.extend([""] * (num_cols - len(values)))
            elif len(values) > num_cols:
                values = values[: num_cols - 1] + [" ".join(values[num_cols - 1 :])]
            unified_rows.append(values)
        return unified_rows

    def clean_table(self, header, rows):
        """
        Returns:
            df: pandas.DataFrame or None
            text_version: fallback text representation
        """
        header = self.clean_header(header)
        num_cols = len(header)
        cleaned_rows = self.unify_rows(rows, num_cols) if num_cols > 0 else []

        try:
            df = pd.DataFrame(cleaned_rows, columns=header)
        except Exception:
            df = None

        lines = []
        if header:
            lines.append(" | ".join(header))
        for row in cleaned_rows:
            lines.append(" | ".join(row))
        text_version = "\n".join(lines)

        return df, text_version
