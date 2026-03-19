using JET

@testset "JET" begin
    @test isempty(
        JET.get_reports(
            JET.report_package(TensorTrainNumerics; target_modules=(TensorTrainNumerics,))
        )
    )
end
