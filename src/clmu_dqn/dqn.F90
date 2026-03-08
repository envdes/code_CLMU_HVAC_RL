! ==============================================================================
! 模块：定义网络结构和参数
! ==============================================================================
module dqn_network_mod
#include "shr_assert.h"

    use shr_kind_mod      , only : r8 => shr_kind_r8
    use clm_varctl        , only : iulog
    implicit none
    save  ! 确保变量在整个模拟期间常驻内存
    logical :: is_loaded = .false.  ! 增加一个状态位，标记模型是否已加载
    ! --- 网络超参数 ---
    integer, parameter :: in_dim = 12
    integer, parameter :: hidden_dim = 128
    integer, parameter :: out_dim = 9
    
    ! --- 网络权重 (静态分配内存) ---
    ! 注意：Fortran 是列优先 (Column-Major)。
    ! PyTorch 的权重形状是 (Out, In)。
    ! 当我们读取二进制流时，Fortran 会自动将数据填充为 (Out, In) 的矩阵。
    ! 因此：w(i, j) 对应 PyTorch 的 weight[i, j]。
    ! 计算时直接使用 matmul(w, x) 即可，无需转置。
    
    real(r8) :: w0(hidden_dim, in_dim), b0(hidden_dim)      ! 第一层
    real(r8) :: w2(hidden_dim, hidden_dim), b2(hidden_dim)  ! 第二层
    real(r8) :: w4(out_dim, hidden_dim), b4(out_dim)        ! 输出层

    ! --- 电价数据存储 ---
    integer :: m, d, i, stat
    real :: h, price
    real, dimension(12, 31, 48) :: price_grid ! 月, 日, 半小时步长(24*2)
    logical :: price_loaded = .false.

contains

    ! --- 子程序：从二进制文件加载权重 ---
    subroutine load_model(filename)
        character(len=*), intent(in) :: filename
        integer :: u, ios
        logical :: file_exists
        if (is_loaded) return  ! 如果已经加载，直接返回
        
        inquire(file=filename, exist=file_exists)
        if (.not. file_exists) then
            print *, "Error: Model file not found: ", filename
            stop
        end if

        !print *, "Loading model from: ", filename
        
        ! 使用 stream 访问模式读取二进制流
        ! open(newunit=u, file=filename, access='stream', status='old', action='read')
        ! 2. 使用 iostat 监控打开状态
        open(newunit=u, file=trim(filename), access='stream', &
             status='old', action='read', iostat=ios, &
             convert='little_endian')
        
        if (ios /= 0) then
            write(iulog,*) "ERROR: Failed to open model file, ios =", ios
            return
        end if

        ! 3. 使用 iostat 监控读取状态
        read(u, iostat=ios) w0, b0
        read(u, iostat=ios) w2, b2
        read(u, iostat=ios) w4, b4
        
        if (ios /= 0) then
            write(iulog,*) "ERROR: Failed to read model parameters, ios =", ios
            close(u)
            return
        end if
        ! 必须严格按照 Python 写入的顺序读取
        !read(u) w0, b0
        !read(u) w2, b2
        !read(u) w4, b4

        write(iulog,*) "CLM_DEBUG in dqnm: r8 kind value =", kind(w0(1,1))
        write(iulog,*) "CLM_DEBUG in dqnm: w0(1,1) =", w0(1,1)

        close(u)
        is_loaded = .true.     ! 标记为已加载
        !print *, "Model loaded successfully."
    end subroutine load_model

    subroutine unload_model()
        ! 重置加载状态
        is_loaded = .false.
        !deallocate(w0, b0, w2, b2, w4, b4)
    end subroutine unload_model

    subroutine load_price_ele(filename)
        character(len=*), intent(in) :: filename

        if (price_loaded) return  ! 如果已经加载，直接返回
        price_loaded = .true.     ! 标记为已加载
        open(unit=10, file=filename, status='old', action='read')
        
        ! 跳过表头 (如果第一行是标题)
        read(10, *) 
        
        do
            ! 假设 CSV 格式为: month,day,hour,price
            ! 使用 comma 作为分隔符读取
            read(10, *, iostat=stat) m, d, h, price
            if (stat /= 0) exit ! 读到文件末尾退出
            
            ! 将 hour 转换为整数索引 (例如 0.5 对应 索引 2)
            i = nint(h * 2) + 1
            price_grid(m, d, i) = price
        end do

        close(10)
    end subroutine load_price_ele
    ! --- 函数：前向推理 (Forward Pass) ---
    function dqnpredict(input_vec) result(output_vec)
        real(r8), intent(in) :: input_vec(in_dim)
        real(r8) :: output_vec(out_dim)
        
        ! 中间层缓冲区
        real(r8) :: h1(hidden_dim)
        real(r8) :: h2(hidden_dim)

        ! === Layer 1: Linear + ReLU ===
        ! Y = W * X + b
        h1 = matmul(w0, input_vec) + b0
        ! ReLU Activation: max(0, x)
        h1 = max(0.0, h1)

        ! === Layer 2: Linear + ReLU ===
        h2 = matmul(w2, h1) + b2
        h2 = max(0.0, h2)
        ! === Layer 3: Linear (Output) ===
        ! 输出层通常没有激活函数 (对于 DQN 是 Q值)
        output_vec = matmul(w4, h2) + b4

    end function dqnpredict

end module dqn_network_mod