__copyright__ = """
Copyright (C) 2020 Cory Mikida
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# avoid spurious: pytest.mark.parametrize is not callable
# pylint: disable=not-callable

import numpy as np
from leap.multistep.multirate import (
        MultiRateHistory as MRHistory,
        EmbeddedMultiRateMultiStepMethodBuilder)


from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


def test_embedded(order=3, step_ratio=2):
    # Solve
    # f' = f+s
    # s' = -f+s

    def true_f(t):
        return np.exp(t)*np.sin(t)

    def true_s(t):
        return np.exp(t)*np.cos(t)

    rtol = 1e-6

    method = EmbeddedMultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(1, "<func>f", ("fast", "slow",)),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(step_ratio, "<func>s", ("fast", "slow"))
                        ),
                    ),
                static_dt=False,
                rtol=rtol)

    code = method.generate()
    #print(code)

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")

    stepper_cls = codegen.get_class(code)

    t = 0
    dt = 2**(-4)
    final_t = 10

    stepper = stepper_cls(
            function_map={
                "<func>f": lambda t, fast, slow: fast + slow,
                "<func>s": lambda t, fast, slow: -fast + slow,
                })

    stepper.set_up(
            t_start=t, dt_start=dt,
            context={
                "fast": true_f(t),
                "slow": true_s(t),
                })

    f_times = []
    f_values = []
    s_times = []
    s_values = []
    fast_times = []
    fast_values = []
    slow_times = []
    slow_values = []
    step_sizes = []
    last_t = 0
    istep = 0
    for event in stepper.run(t_end=final_t):
        if isinstance(event, stepper_cls.StateComputed):
            if event.component_id == "fast":
                f_times.append(event.t)
                f_values.append(event.state_component)
            elif event.component_id == "slow":
                s_times.append(event.t)
                s_values.append(event.state_component)
            else:
                assert False, event.component_id
        elif isinstance(event, stepper_cls.StepCompleted):
            if not f_times:
                continue

            # Account for bootstrapping.
            if istep < 3:
                step_sizes.append((event.t - last_t)/2.0)
                step_sizes.append((event.t - last_t)/2.0)
            else:
                step_sizes.append(event.t - last_t)
            istep += 1
            last_t = event.t
            slow_times.extend(s_times)
            fast_times.extend(f_times)
            slow_values.extend(s_values)
            fast_values.extend(f_values)
            del s_times[:]
            del f_times[:]
            del s_values[:]
            del f_values[:]
        elif isinstance(event, stepper_cls.StepFailed):
            print("Step failed at t = ", event.t)
            del s_times[:]
            del f_times[:]
            del s_values[:]
            del f_values[:]

    fast_times = np.array(fast_times)
    slow_times = np.array(slow_times)
    fast_values_true = true_f(fast_times)
    slow_values_true = true_s(slow_times)

    #f_err = fast_values - fast_values_true
    #s_err = slow_values - slow_values_true
    f_err_rel = (fast_values - fast_values_true)/fast_values_true
    s_err_rel = (slow_values - slow_values_true)/slow_values_true

    #print(slow_values)
    #print(slow_values_true)
    #print(s_err_rel)

    step_sizes = np.array(step_sizes)

    import matplotlib.pyplot as pt
    pt.plot(slow_times, slow_values_true)
    pt.title("True Slow Solution")
    pt.xlabel("t")
    pt.ylabel("s")
    pt.show()
    #pt.plot(slow_times, s_err)
    #pt.show()
    pt.plot(slow_times, np.log10(abs(s_err_rel)))
    pt.plot(slow_times, -6*np.ones(len(slow_times)), 'r-')
    pt.title("Relative Error: Slow Solution")
    pt.xlabel("t")
    pt.ylabel("Relative Error")
    pt.show()
    pt.plot(fast_times, np.log10(abs(f_err_rel)))
    pt.show()
    pt.plot(slow_times, step_sizes)
    pt.title("Timestep Sizes")
    pt.xlabel("t")
    pt.ylabel("dt")
    pt.show()

    assert max(s_err_rel) <= rtol
    assert max(f_err_rel) <= rtol


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
