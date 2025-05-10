! gfortran -c sac_actor.f90 -I/usr/lib64/gfortran/modules
! gfortran -o sac_program sac.f90 sac_actor.o -I/usr/lib64/gfortran/modules -L/usr/lib64 -lnetcdff

module actor_module
    use netcdf
    implicit none
    real(8), dimension(:,:), allocatable :: fc1_weights, fc2_weights, fc_mean_weights, fc_logstd_weights
    real(8), dimension(:), allocatable :: fc1_bias, fc2_bias, fc_mean_bias, fc_logstd_bias
    real(8), dimension(:), allocatable :: action_scale, action_bias
    real(8), parameter :: LOG_STD_MIN = -5_8, LOG_STD_MAX = 2.0_8
    real(8), parameter :: pi = 3.14159265358979323846_8
  
    private
    
    !---------------------------------------------------------------------
    ! public interfaces
    public  :: get_action, nn_cf_net_init, nn_cf_net_finalize
  
  contains
  
  ! ---------------------------activation functions----------------------
    ! 激活函数ReLU
    real(8) function relu(x)
        real(8), intent(in) :: x
        if (x > 0.0_8) then
            relu = x
        else
            relu = 0.0_8
        end if
    end function relu
  
    ! Tanh of activation function
    real(8) function tanh_f(x)
        real(8), intent(in) :: x
        tanh_f = tanh(x)
    end function tanh_f
  ! ---------------------------------------------------------------------
  
  ! --------------------------forward and get_action---------------------
    ! pytorch for forward function
    subroutine forward(x, mean, log_std)
        real(8), dimension(:), intent(in) :: x
        real(8), dimension(:), intent(out) :: mean, log_std
        real(8), dimension(size(fc1_bias)) :: h1
        real(8), dimension(size(fc2_bias)) :: h2
  
        print *, "fc1_weights shape: ", shape(fc1_weights)
        ! first layer full connection + activation
        !h1 = matmul(fc1_weights, x) + fc1_bias
        h1 = matmul(x, fc1_weights) + fc1_bias
        h1 = apply_relu(h1)
  
        ! second layer full connection + activation
        !h2 = matmul(fc2_weights, h1) + fc2_bias
        h2 = matmul(h1, fc2_weights) + fc2_bias
        h2 = apply_relu(h2)
  
        print *, "fc_mean_weights shape: ", shape(fc_mean_weights)
        ! mean and log_std
        !mean = matmul(fc_mean_weights, h2) + fc_mean_bias
        mean = matmul(h2, fc_mean_weights) + fc_mean_bias
        print *, "mean shape: ", shape(mean)
  
        !log_std = matmul(fc_logstd_weights, h2) + fc_logstd_bias
        log_std = matmul(h2, fc_logstd_weights) + fc_logstd_bias
        !log_std = tanh_f(log_std)
        print *, "log_std shape: ", shape(log_std)
        log_std = apply_tanh(log_std)
        log_std = LOG_STD_MIN + 0.5_8 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0_8)
    end subroutine forward
  
    ! get action and log_prob from pytorch sac version
    subroutine get_action(x, action, log_prob, mean)
        real(8), dimension(:), intent(in) :: x
        real(8), dimension(:), intent(out) :: action, log_prob, mean
        !real(8), dimension(:) :: mean_val, log_std, std, x_t, y_t
        real(8), dimension(size(fc_mean_bias)) :: mean_val
        real(8), dimension(size(fc_logstd_bias)) :: log_std
        real(8), dimension(size(fc_logstd_bias)) :: std
        real(8), dimension(size(fc_logstd_bias)) :: x_t
        real(8), dimension(size(fc_logstd_bias)) :: y_t
  
        call forward(x, mean_val, log_std)
  
        std = exp(log_std)
        call normal_distribution(mean_val, std, x_t)
        !y_t = tanh_f(x_t)
        y_t = apply_tanh(x_t)
  
        ! get action
        action = y_t * action_scale + action_bias
  
        ! get log_prob
        log_prob = log_probability(x_t, y_t, mean_val, std)
        
        ! calculate mean
        !mean = tanh_f(mean_val) * action_scale + action_bias
        mean = apply_tanh(mean_val) * action_scale + action_bias
    end subroutine get_action
  
    ! Apply ReLU to an array
    function apply_relu(x) result(y)
        real(8), dimension(:), intent(in) :: x
        real(8), dimension(size(x)) :: y
        integer :: i
        do i = 1, size(x)
            y(i) = relu(x(i))
        end do
    end function apply_relu
  
    ! Apply tanh to an array
      function apply_tanh(x) result(y)
          real(8), dimension(:), intent(in) :: x
          real(8), dimension(size(x)) :: y
          integer :: i
          do i = 1, size(x)
              y(i) = tanh_f(x(i))
          end do
      end function apply_tanh
  
  ! ---------------------------normal distribution------------------------
  subroutine normal_distribution(mean, std, sample)
      real(8), dimension(:), intent(in) :: mean, std
      real(8), dimension(:), intent(out) :: sample
      integer :: i
      integer :: n
      ! get sample size
      n = size(sample)
      ! get random normal distribution
      do i = 1, n
          sample(i) = random_normal()  ! get the standard normal distribution
      end do
      ! return the normal distribution
      sample = mean + std * sample
  end subroutine normal_distribution
  
    ! generate standard normal distribution
    real(8) function random_normal()
    real(8) :: u1, u2, z0
  
    ! use Box-Muller method to generate standard normal distribution
    call random_number(u1)  ! get the uniform distribution
    call random_number(u2)
    z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
    ! return the standard normal distribution
    random_normal = z0 
  end function random_normal
  
    ! calculate the log probability
    real(8) function log_probability(x_t, y_t, mean, std)
        real(8), dimension(:), intent(in) :: x_t, y_t, mean, std
        log_probability = sum(-0.5_8 * log(2.0_8 * pi * std**2) - ((x_t - mean)**2 / (2.0_8 * std**2)))
        log_probability = log_probability - sum(log(action_scale * (1.0_8 - y_t**2) + 1e-6_8))
    end function log_probability
  
  ! ---------------------------init and finalize--------------------------
    subroutine nn_cf_net_init()
        !! Initialise the neural net
  
        integer :: ncid
        integer :: varid_action_scale, varid_action_bias
        integer :: varid_fc1_weight, varid_fc1_bias
        integer :: varid_fc2_weight, varid_fc2_bias
        integer :: varid_fc_mean_weight, varid_fc_mean_bias
        integer :: varid_fc_logstd_weight, varid_fc_logstd_bias
        integer :: dim_action_scale_0, dim_action_scale_1
        integer :: dim_action_bias_0, dim_action_bias_1
        integer :: dim_fc1_weight_0, dim_fc1_weight_1, dim_fc1_bias_0
        integer :: dim_fc2_weight_0, dim_fc2_weight_1, dim_fc2_bias_0
        integer :: dim_fc_mean_weight_0, dim_fc_mean_weight_1, dim_fc_mean_bias_0
        integer :: dim_fc_logstd_weight_0, dim_fc_logstd_weight_1, dim_fc_logstd_bias_0
        integer :: n_action_scale_0, n_action_scale_1
        integer :: n_action_bias_0, n_action_bias_1
        integer :: n_fc1_weight_0, n_fc1_weight_1, n_fc1_bias_0
        integer :: n_fc2_weight_0, n_fc2_weight_1, n_fc2_bias_0
        integer :: n_fc_mean_weight_0, n_fc_mean_weight_1, n_fc_mean_bias_0
        integer :: n_fc_logstd_weight_0, n_fc_logstd_weight_1, n_fc_logstd_bias_0
  
        ! open the model.nc file
        !print *, "Opening model.nc"
        call check(nf90_open('/p/clmuapp/model.nc', nf90_nowrite, ncid))
    
        ! get the dimension IDs
        !print *, "Getting dimension IDs"
        call check(nf90_inq_dimid(ncid, 'dim_action_scale_0', dim_action_scale_0))
        call check(nf90_inq_dimid(ncid, 'dim_action_scale_1', dim_action_scale_1))
      
        call check(nf90_inq_dimid(ncid, 'dim_action_bias_0', dim_action_bias_0))
        call check(nf90_inq_dimid(ncid, 'dim_action_bias_1', dim_action_bias_1))
    
        call check(nf90_inq_dimid(ncid, 'dim_fc1.weight_0', dim_fc1_weight_0))
        call check(nf90_inq_dimid(ncid, 'dim_fc1.weight_1', dim_fc1_weight_1))
        call check(nf90_inq_dimid(ncid, 'dim_fc1.bias_0', dim_fc1_bias_0))
    
        call check(nf90_inq_dimid(ncid, 'dim_fc2.weight_0', dim_fc2_weight_0))
        call check(nf90_inq_dimid(ncid, 'dim_fc2.weight_1', dim_fc2_weight_1))
        call check(nf90_inq_dimid(ncid, 'dim_fc2.bias_0', dim_fc2_bias_0))
    
        call check(nf90_inq_dimid(ncid, 'dim_fc_mean.weight_0', dim_fc_mean_weight_0))
        call check(nf90_inq_dimid(ncid, 'dim_fc_mean.weight_1', dim_fc_mean_weight_1))
        call check(nf90_inq_dimid(ncid, 'dim_fc_mean.bias_0', dim_fc_mean_bias_0))
    
        call check(nf90_inq_dimid(ncid, 'dim_fc_logstd.weight_0', dim_fc_logstd_weight_0))
        call check(nf90_inq_dimid(ncid, 'dim_fc_logstd.weight_1', dim_fc_logstd_weight_1))
        call check(nf90_inq_dimid(ncid, 'dim_fc_logstd.bias_0', dim_fc_logstd_bias_0))
    
        ! get the lengths of the dimensions
        !print *, "Getting variable lengths"
        call check(nf90_inquire_dimension(ncid, dim_action_scale_0, len=n_action_scale_0))
        call check(nf90_inquire_dimension(ncid, dim_action_scale_1, len=n_action_scale_1))
    
        call check(nf90_inquire_dimension(ncid, dim_action_bias_0, len=n_action_bias_0))
        call check(nf90_inquire_dimension(ncid, dim_action_bias_1, len=n_action_bias_1))
    
        call check(nf90_inquire_dimension(ncid, dim_fc1_weight_0, len=n_fc1_weight_0))
        call check(nf90_inquire_dimension(ncid, dim_fc1_weight_1, len=n_fc1_weight_1))
        call check(nf90_inquire_dimension(ncid, dim_fc1_bias_0, len=n_fc1_bias_0))
    
        call check(nf90_inquire_dimension(ncid, dim_fc2_weight_0, len=n_fc2_weight_0))
        call check(nf90_inquire_dimension(ncid, dim_fc2_weight_1, len=n_fc2_weight_1))
        call check(nf90_inquire_dimension(ncid, dim_fc2_bias_0, len=n_fc2_bias_0))
    
        call check(nf90_inquire_dimension(ncid, dim_fc_mean_weight_0, len=n_fc_mean_weight_0))
        call check(nf90_inquire_dimension(ncid, dim_fc_mean_weight_1, len=n_fc_mean_weight_1))
        call check(nf90_inquire_dimension(ncid, dim_fc_mean_bias_0, len=n_fc_mean_bias_0))
    
        call check(nf90_inquire_dimension(ncid, dim_fc_logstd_weight_0, len=n_fc_logstd_weight_0))
        call check(nf90_inquire_dimension(ncid, dim_fc_logstd_weight_1, len=n_fc_logstd_weight_1))
        call check(nf90_inquire_dimension(ncid, dim_fc_logstd_bias_0, len=n_fc_logstd_bias_0))
  
        ! allocate memory
        ! the shape of fc1_weights is (256, 5) is different from the data in model.nc
        !print *, "Allocating memory"
        allocate(fc1_weights(n_fc1_weight_1, n_fc1_weight_0))
        allocate(fc1_bias(n_fc1_bias_0))
  
        allocate(fc2_weights(n_fc2_weight_1, n_fc2_weight_0))
        allocate(fc2_bias(n_fc2_bias_0))
  
        allocate(fc_mean_weights(n_fc_mean_weight_1, n_fc_mean_weight_0))
        allocate(fc_mean_bias(n_fc_mean_bias_0))
  
        allocate(fc_logstd_weights(n_fc_logstd_weight_1, n_fc_logstd_weight_0))
        allocate(fc_logstd_bias(n_fc_logstd_bias_0))
  
        !allocate(action_scale(n_action_scale_0, n_action_scale_1))
        !allocate(action_bias(n_action_bias_0, n_action_bias_1))
        allocate(action_scale(n_action_scale_1))
        allocate(action_bias(n_action_bias_1))
  
        !print *, "shape of fc1_weights: ", shape(fc1_weights)
  
        ! read the weights and biases
        !print *, "Reading weights and biases"
        call check(nf90_inq_varid(ncid, 'fc1.weight', varid_fc1_weight))
        !print *, "varid_fc1_weight: ", varid_fc1_weight
        call check(nf90_get_var(ncid, varid_fc1_weight, fc1_weights))
        !print *, "fc1_weights shape: ", shape(fc1_weights)
        call check(nf90_inq_varid(ncid, 'fc1.bias', varid_fc1_bias))
        call check(nf90_get_var(ncid, varid_fc1_bias, fc1_bias))
        !print *, "fc1_bias shape: ", shape(fc1_bias)
  
        call check(nf90_inq_varid(ncid, 'fc2.weight', varid_fc2_weight))
        call check(nf90_get_var(ncid, varid_fc2_weight, fc2_weights))
        call check(nf90_inq_varid(ncid, 'fc2.bias', varid_fc2_bias))
        call check(nf90_get_var(ncid, varid_fc2_bias, fc2_bias))
  
        call check(nf90_inq_varid(ncid, 'fc_mean.weight', varid_fc_mean_weight))
        call check(nf90_get_var(ncid, varid_fc_mean_weight, fc_mean_weights))
        call check(nf90_inq_varid(ncid, 'fc_mean.bias', varid_fc_mean_bias))
        call check(nf90_get_var(ncid, varid_fc_mean_bias, fc_mean_bias))
  
        call check(nf90_inq_varid(ncid, 'fc_logstd.weight', varid_fc_logstd_weight))
        call check(nf90_get_var(ncid, varid_fc_logstd_weight, fc_logstd_weights))
        call check(nf90_inq_varid(ncid, 'fc_logstd.bias', varid_fc_logstd_bias))
        call check(nf90_get_var(ncid, varid_fc_logstd_bias, fc_logstd_bias))
  
        !print *, "getting action scale and bias"
        call check(nf90_inq_varid(ncid, 'action_scale', varid_action_scale))
        call check(nf90_get_var(ncid, varid_action_scale, action_scale))
        call check(nf90_inq_varid(ncid, 'action_bias', varid_action_bias))
        call check(nf90_get_var(ncid, varid_action_bias, action_bias))
  
        ! close the model.nc file
        !print *, "Model loaded successfully"
        call check(nf90_close(ncid))
        !print *, "Model loaded"
    end subroutine nn_cf_net_init
  
    subroutine nn_cf_net_finalize()
        !! 释放神经网络的内存
        if (allocated(fc1_weights)) deallocate(fc1_weights)
        if (allocated(fc1_bias)) deallocate(fc1_bias)
        if (allocated(fc2_weights)) deallocate(fc2_weights)
        if (allocated(fc2_bias)) deallocate(fc2_bias)
        if (allocated(fc_mean_weights)) deallocate(fc_mean_weights)
        if (allocated(fc_mean_bias)) deallocate(fc_mean_bias)
        if (allocated(fc_logstd_weights)) deallocate(fc_logstd_weights)
        if (allocated(fc_logstd_bias)) deallocate(fc_logstd_bias)
        if (allocated(action_scale)) deallocate(action_scale)
        if (allocated(action_bias)) deallocate(action_bias)
    end subroutine nn_cf_net_finalize
  
    subroutine check(ierr)
        integer, intent(in) :: ierr
        if (ierr /= nf90_noerr) then
            print *, "Error: ", ierr
            stop
        end if
    end subroutine check
  
  end module actor_module
  