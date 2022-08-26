#include "core/providers/pim/pim_provider_factory.h"

#include <memory>

#include "gsl/gsl"

#include "core/common/make_unique.h"
#include "core/providers/pim/pim_execution_provider.h"
#include "core/providers/pim/pim_execution_provider_info.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct PIMProviderFactory : IExecutionProviderFactory {
  PIMProviderFactory(const PIMExecutionProviderInfo& info)
    : info_{info} {}
  ~PIMProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

private:
  PIMExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> PIMProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<PIMExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_PIM(const PIMExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::PIMProviderFactory>(info);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_PIM,
                    _In_ OrtSessionOptions* options, int device_id) {
    PIMExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(device_id);
    options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_PIM(info));

  return nullptr;
}

