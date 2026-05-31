
1. calc.py:
可以直接copy ../sed.py，但是我希望只暴露：
compute_dipole_sed
DipoleSedResult
generate_commensurate_qpath
save_result_npz

2. 此外，save_result_npz rename为save_sed; compute_dipole_sed rename为calculate_sed

3. calc里面添加一个load_sed（从npz文件load一个DipoleSedResult）

3. 对plot：只需要plot_sed function暴露，删除load_sed, plot_sed_file；

4. __init__.py是核心，实际使用的时候，都从这里导入

例如：from ferrodispcalc.sed import calculate_sed, load_sed, plot_sed, save_sed

5. 我不需要程序自动生成qpath，用户得自己使用generate_commensurate_qpath；