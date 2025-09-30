#!/usr/bin/env bash
# grid_search.sh
#
# 这个脚本会遍历以下几个超参数组合
#   - 学习率 (lr):                1e-4   1e-5
#   - Batch size:                4      8
#   - 序列长度 (max_seq):         9
#   - Transformer 隐藏维度 (hidden_dim): 512  1024
#   - 交叉熵 Loss 权重 (w_ce):       1.0    1.0  （一般固定为 1）
#   - Dice Loss 权重 (w_dice):      2
#   - Jaccard Loss 权重 (w_jaccard): 2
#   - 形状单调惩罚权重 (w_shape_mono): 0.5 0.5  （可根据需要调整）
#   - 深度单调惩罚权重 (w_depth_mono): 0.5 0.5
#   - 水平扩展膨胀权重 (w_expand):    2.0  2.0  （可根据需要调整）
#
# 对于每一个组合，会调用 python transformer_unet.py train 进行训练与验证，
# 并把输出日志、最终模型和可视化结果保存到对应子目录中。

########## 用户需要修改的部分 ##########
# python 可执行文件
PYTHON_EXEC=python

# 你的训练脚本，已按照新版接口修改
# TRAIN_SCRIPT=transformer_unet.py
TRAIN_SCRIPT=U-T.py
# 要遍历的多个数据目录
declare -a DATA_DIRS=("data")

# 训练轮数（一般保持一致）
EPOCHS=50
########## 结束 ##########

# 超参数列表
declare -a LRS=(1e-4)
declare -a BATCH_SIZES=(4)
declare -a MAX_SEQS=(9)
declare -a HIDDENS=(1024)
declare -a W_CES=(1.0)             # 交叉熵 Loss 权重
declare -a W_DICES=(1)
declare -a W_JACCARDS=(1)
declare -a W_SHAPE_MONOS=(0)
declare -a W_DEPTH_MONOS=(10)
declare -a W_EXPANDS=(50)

# 对每个数据目录分别执行网格搜索
for DATA_DIR in "${DATA_DIRS[@]}"; do

  # 保存网格搜索结果的顶层输出目录
  OUTPUT_ROOT="./${DATA_DIR}_hyperparam_search_results"
  mkdir -p "${OUTPUT_ROOT}"

  # 遍历所有组合
  for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      for MS in "${MAX_SEQS[@]}"; do
        for HD in "${HIDDENS[@]}"; do
          for WCE in "${W_CES[@]}"; do
            for WD in "${W_DICES[@]}"; do
              for WJ in "${W_JACCARDS[@]}"; do
                for WSM in "${W_SHAPE_MONOS[@]}"; do
                  for WDM in "${W_DEPTH_MONOS[@]}"; do
                    for WE in "${W_EXPANDS[@]}"; do

                      # 子目录命名：lr_x_bs_x_seq_x_hd_x_wCE_x_wD_x_wJ_x_wSM_x_wDM_x_wE_x
                      SUBDIR="lr_${LR}_bs_${BS}_seq_${MS}_hd_${HD}_wCE_${WCE}_wD_${WD}_wJ_${WJ}_wSM_${WSM}_wDM_${WDM}_wE_${WE}"
                      EXP_DIR="${OUTPUT_ROOT}/${SUBDIR}"
                      mkdir -p "${EXP_DIR}"

                      # 日志、模型和可视化目录
                      LOG_FILE="${EXP_DIR}/train_log.txt"
                      MODEL_DIR="${EXP_DIR}/models"
                      RESULTS_DIR="${EXP_DIR}/results"
                      mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"

                      echo "=============================================="
                      echo " Starting experiment (data dir: ${DATA_DIR}):"
                      echo "   LR              = ${LR}"
                      echo "   BatchSize       = ${BS}"
                      echo "   MaxSeq          = ${MS}"
                      echo "   HiddenDim       = ${HD}"
                      echo "   w_ce            = ${WCE}"
                      echo "   w_dice          = ${WD}"
                      echo "   w_jaccard       = ${WJ}"
                      echo "   w_shape_mono    = ${WSM}"
                      echo "   w_depth_mono    = ${WDM}"
                      echo "   w_expand        = ${WE}"
                      echo "   DataDir         = ${DATA_DIR}"
                      echo "   OutputDir       = ${EXP_DIR}"
                      echo "=============================================="
                      echo ""

                      # 调用训练脚本（加上子命令 train）
                      ${PYTHON_EXEC} "${TRAIN_SCRIPT}" train \
                        --data_dir        "${DATA_DIR}" \
                        --epochs          "${EPOCHS}" \
                        --batch           "${BS}" \
                        --lr              "${LR}" \
                        --max_seq         "${MS}" \
                        --hidden_dim      "${HD}" \
                        --w_ce            "${WCE}" \
                        --w_dice          "${WD}" \
                        --w_jaccard       "${WJ}" \
                        --w_shape_mono    "${WSM}" \
                        --w_depth_mono    "${WDM}" \
                        --w_expand        "${WE}" \
                        --results_dir     "${RESULTS_DIR}" \
                        --model_out_dir   "${MODEL_DIR}" \
                        --log_out         "${LOG_FILE}" \
                        2>&1 | tee "${LOG_FILE}"

                      echo ""
                      echo " Finished: ${SUBDIR} (data dir: ${DATA_DIR})"
                      echo ""

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done

  echo "All experiments for ${DATA_DIR} completed. Results under ${OUTPUT_ROOT}/"
  echo ""
done

echo "All data directories done!" 
