import glob
import os

#フォルダ内ファイル名一括取得関数
def file_search(input_dir, output_dir):
    path_list = glob.glob(input_dir + '\\*') #指定されたディレクトリ内のすべてのファイルを取得
    name_list = [] #ファイル名の空リスト
    ext_list = [] #拡張子の空リスト
    out_list = [] #保存パスの空リスト

    #ファイルのフルパスからファイル名と拡張子を抽出
    for i in path_list:
        file = os.path.basename(i) #ファイル名を取得
        name, ext = os.path.splitext(file) #ファイル名と拡張子を分割
        name_list.append(name)
        ext_list.append(ext)
        out_list.append(os.path.join(*[output_dir, name + '_out' + ext])) #保存パスを作成
    return path_list, name_list, ext_list, out_list
