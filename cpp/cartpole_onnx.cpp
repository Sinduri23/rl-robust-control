// Closed-loop control in C++: CartPole physics + the trained policy (ONNX Runtime).
//
// This mirrors how a learned controller runs on a real machine: no Python, a fixed
// control loop, and a measured inference time per step.
//
// Build: see cpp/CMakeLists.txt or README.  Run: ./cartpole_onnx ../models/policy.onnx [episodes] [mass_scale]
#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ---- CartPole physics: same equations and constants as Gymnasium CartPole-v1 ----
struct CartPole {
    double gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5, force_mag = 10.0, tau = 0.02;
    double x = 0, x_dot = 0, theta = 0, theta_dot = 0;

    void reset(std::mt19937& rng) {
        std::uniform_real_distribution<double> u(-0.05, 0.05);
        x = u(rng); x_dot = u(rng); theta = u(rng); theta_dot = u(rng);
    }

    // returns true if the episode is over (pole fell or cart left the track)
    bool step(int action) {
        const double total_mass = masspole + masscart;
        const double polemass_length = masspole * length;
        const double force = action == 1 ? force_mag : -force_mag;
        const double c = std::cos(theta), s = std::sin(theta);
        const double temp = (force + polemass_length * theta_dot * theta_dot * s) / total_mass;
        const double theta_acc = (gravity * s - c * temp) /
                                 (length * (4.0 / 3.0 - masspole * c * c / total_mass));
        const double x_acc = temp - polemass_length * theta_acc * c / total_mass;
        x += tau * x_dot;           x_dot += tau * x_acc;          // explicit Euler, as in Gymnasium
        theta += tau * theta_dot;   theta_dot += tau * theta_acc;
        return std::fabs(x) > 2.4 || std::fabs(theta) > 12.0 * M_PI / 180.0;
    }

    std::array<float, 4> obs() const {
        return {static_cast<float>(x), static_cast<float>(x_dot),
                static_cast<float>(theta), static_cast<float>(theta_dot)};
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s policy.onnx [episodes=20] [mass_scale=1.0]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const int episodes = argc > 2 ? std::atoi(argv[2]) : 20;
    const double mass_scale = argc > 3 ? std::atof(argv[3]) : 1.0;

    // ---- load the policy ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cartpole");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);   // single thread: predictable latency, like an embedded target
    Ort::Session session(env, model_path.c_str(), opts);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {"obs"};
    const char* out_names[] = {"logits"};
    const std::array<int64_t, 2> shape{1, 4};

    std::mt19937 rng(123);
    CartPole sim;
    sim.masspole *= mass_scale;

    long total_steps = 0;
    double total_us = 0, worst_us = 0;
    int successes = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        sim.reset(rng);
        int t = 0;
        bool done = false;
        while (!done && t < 500) {
            std::array<float, 4> o = sim.obs();
            auto t0 = std::chrono::high_resolution_clock::now();
            Ort::Value input = Ort::Value::CreateTensor<float>(mem, o.data(), o.size(), shape.data(), shape.size());
            auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
            const float* logits = out[0].GetTensorData<float>();
            const int action = logits[1] > logits[0] ? 1 : 0;   // argmax over {left, right}
            double us = std::chrono::duration<double, std::micro>(
                            std::chrono::high_resolution_clock::now() - t0).count();
            total_us += us;
            if (us > worst_us) worst_us = us;
            done = sim.step(action);
            ++t;
        }
        total_steps += t;
        successes += (t >= 500);
    }

    std::printf("episodes          : %d (pole mass x%.1f)\n", episodes, mass_scale);
    std::printf("mean steps        : %.1f / 500\n", double(total_steps) / episodes);
    std::printf("success rate      : %.0f%%\n", 100.0 * successes / episodes);
    std::printf("inference latency : mean %.1f us, worst %.1f us  (control period is 20000 us)\n",
                total_us / total_steps, worst_us);
    return 0;
}
