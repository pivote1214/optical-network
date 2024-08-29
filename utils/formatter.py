# import openpyxl
# from openpyxl.utils import get_column_letter
# from openpyxl.worksheet.worksheet import Worksheet
# from openpyxl.styles import Font, Border, Side, Alignment


# def format_excel(
#     excel_file_path: str | openpyxl.Workbook, 
#     font_size: int = 14, 
#     font_name: str = 'Meiryo UI', 
#     header: bool = True, 
#     width_list: dict[int, float] | list = None, 
#     height: float = None, 
#     alignment_horrizonal: dict[int, str] | list = None, 
#     sheet_title: bool = False
#     ) -> None | openpyxl.Workbook:
#     """
#     エクセルファイルのフォーマットを整える
#     """
#     # エクセルファイルをロード
#     if isinstance(excel_file_path, str):
#         workbook = openpyxl.load_workbook(excel_file_path)
#     else:
#         workbook = excel_file_path
    
#     for sheet in workbook.worksheets:
#         for i, row in enumerate(sheet.iter_rows()):
#             for cell in row:
#                 if header and i == 0:
#                     cell.font = Font(name=font_name, size=font_size, bold=True)
#                 else:
#                     cell.font = Font(name=font_name, size=font_size)
#                 cell.alignment = Alignment(horizontal='center', vertical='center')

#         if sheet_title:
#             # 1行目にシート名を追加
#             sheet.insert_rows(0)
#             sheet.cell(row=1, column=1).value = sheet.title
#             sheet.cell(row=1, column=1).font = Font(name=font_name, size=font_size, bold=True)
#             # データの範囲をマージ
#             sheet.merge_cells(start_row=1, start_column=1, 
#                               end_row=1, end_column=sheet.max_column)
#             sheet.cell(row=1, column=1).alignment = Alignment(horizontal='center', vertical='center')
#             # 罫線を調整
#             _adjust_borders(sheet, sheet_title=True)
#         else:
#             _adjust_borders(sheet, sheet_title=False)

#         if width_list is not None:
#             if isinstance(width_list, list):
#                 width_list = {i+1: width for i, width in enumerate(width_list)}
#             for j, width in width_list.items():
#                 column_char = get_column_letter(j)
#                 sheet.column_dimensions[column_char].width = width

#         if height is not None:
#             for i in range(1, sheet.max_row+1):
#                 sheet.row_dimensions[i].height = height

#         if alignment_horrizonal is not None:
#             if isinstance(alignment_horrizonal, list):
#                 alignment_horrizonal = {i+1: pos for i, pos in enumerate(alignment_horrizonal)}
#             for j, pos in alignment_horrizonal.items():
#                 for i in range(2, sheet.max_row+1):
#                     sheet.cell(row=i, column=j).alignment = Alignment(horizontal=pos, vertical='center')
 
#     # エクセルファイルを保存
#     if isinstance(excel_file_path, str):
#         workbook.save(excel_file_path)
#     else:
#         return workbook


# def _adjust_borders(
#     ws: Worksheet, 
#     sheet_title: bool
#     ) -> Worksheet:
#     """
#     外枠は太い罫線，それ以外は細い罫線でExcelファイルを整形
#     """
#     thin = Side(border_style="thin", color="000000")
#     medium = Side(border_style="medium", color="000000")
#     thick = Side(border_style="thick", color="000000")
    
#     # ワークシートの最大行と最大列を取得
#     max_row = ws.max_row
#     max_col = ws.max_column
    
#     for i, row in enumerate(ws.iter_rows()):
#         for j, cell in enumerate(row):
#             if i == 0:  # Top border
#                 border_top = thick
#             elif i == 1 or (i == 2 and sheet_title):
#                 border_top = medium
#             else:
#                 border_top = thin
#             if i == max_row - 1:  # Bottom border
#                 border_bottom = thick
#             else:
#                 border_bottom = thin
#             if j == 0:  # Left border
#                 border_left = thick
#             else:
#                 border_left = thin
#             if j == max_col - 1:  # Right border
#                 border_right = thick
#             else:
#                 border_right = thin
            
#             cell.border = Border(
#                 top=border_top, 
#                 bottom=border_bottom, 
#                 left=border_left, 
#                 right=border_right
#             )

#     return ws
