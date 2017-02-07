#include <boost/filesystem.hpp>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/TaQL/TableParse.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ColumnDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableColumn.h>
#include "purify/casacore.h"
#include "purify/directories.h"

#include "purify/types.h"
#include "purify/utilities.h"

#include "catch.hpp"
using namespace ::casacore;
using namespace purify::notinstalled;
TEST_CASE("Casacore") {
  // create the table descriptor
  casa::TableDesc simpleDesc = casa::MS::requiredTableDesc();
  // set up a new table
  casa::SetupNewTable newTab("simpleTab", simpleDesc, casa::Table::New);
  // create the MeasurementSet
  casa::MeasurementSet simpleMS(newTab);
  // now we need to define all required subtables
  // the following call does this for us if we don't need to
  // specify details of Storage Managers for columns.
  simpleMS.createDefaultSubtables(casa::Table::New);
  // fill MeasurementSet via its Table interface
  // For example, construct one of the columns
  casa::TableColumn feed(simpleMS, casa::MS::columnName(casa::MS::FEED1));
  casa::uInt rownr = 0;
  // add a row
  simpleMS.addRow();
  // set the values in that row, e.g. the feed column
  feed.putScalar(rownr, 1);
  // Access a subtable
  casa::ArrayColumn<casa::Double> antpos(simpleMS.antenna(),
      casa::MSAntenna::columnName(casa::MSAntenna::POSITION));
  simpleMS.antenna().addRow();
  casa::Array<casa::Double> position(casa::IPosition(1, 3));
  position(casa::IPosition(1, 0)) = 1.;
  position(casa::IPosition(1, 1)) = 2.;
  position(casa::IPosition(1, 2)) = 3.;
  antpos.put(0, position);
}

class TmpPath {
  public:
    TmpPath()
      : path_(boost::filesystem::unique_path(boost::filesystem::temp_directory_path()
            / "%%%%-%%%%-%%%%-%%%%.ms")) {}
    ~TmpPath() {
      if(boost::filesystem::exists(path()))
        boost::filesystem::remove_all(path());
    }
    boost::filesystem::path const &path() const { return path_; }

  private:
    boost::filesystem::path path_;
};

class TmpMS : public TmpPath {
  public:
    TmpMS() : TmpPath() {
      casa::TableDesc simpleDesc = casa::MS::requiredTableDesc();
      casa::SetupNewTable newTab(path().string(), simpleDesc, casa::Table::New);
      ms_.reset(new casa::MeasurementSet(newTab));
      ms_->createDefaultSubtables(casa::Table::New);
    }
    casa::MeasurementSet const &operator*() const { return *ms_; }
    casa::MeasurementSet &operator*() { return *ms_; }

    casa::MeasurementSet const *operator->() const { return ms_.get(); }
    casa::MeasurementSet *operator->() { return ms_.get(); }

  protected:
    std::unique_ptr<casa::MeasurementSet> ms_;
};

TEST_CASE("Size/Number of channels") {
  CHECK(purify::casa::MeasurementSet(purify::notinstalled::ngc5921_2_ms()).size() == 63);
}

TEST_CASE("Single channel") {
  namespace pc = purify::casa;
  auto const ms = pc::MeasurementSet(purify::notinstalled::ngc5921_2_ms());
  SECTION("Check channel validity") {
    //CHECK(not pc::MeasurementSet::const_iterator::value_type(0, ms).is_valid());
    CHECK(pc::MeasurementSet::const_iterator::value_type(17, ms).is_valid());
  }
  SECTION("Raw UVW") {
    auto const channel = pc::MeasurementSet::const_iterator::value_type(17, ms);
    REQUIRE(channel.size() == 12852);
    auto const u = channel.raw_u();
    REQUIRE(u.size() == 12852);
    CHECK(std::abs(u[0]) < 1e-8);
    CHECK(std::abs(u[10000] - 81.7386695336412145707072340883314609527587890625) < 1e-8);
    auto const v = channel.raw_v();
    REQUIRE(v.size() == 12852);
    CHECK(std::abs(v[0]) < 1e-8);
    CHECK(std::abs(v[10000] - -22.980947660874132765229660435579717159271240234375) < 1e-8);
  }

  SECTION("Raw frequencies") {
    auto const f0 = pc::MeasurementSet::const_iterator::value_type(0, ms).raw_frequencies();
    CHECK(f0.size() == 1);
    CHECK(std::abs(f0(0) - 1412665073.768775463104248046875) < 1e-1);
    //CHECK(std::abs(f0(1) - 111450812500.10001) < 1e-4);

    auto const f17 = pc::MeasurementSet::const_iterator::value_type(17, ms).raw_frequencies();
    CHECK(f17.size() == 1);
    CHECK(std::abs(f17(0) - 1413080112.831275463104248046875) < 1e-1);
    //CHECK(std::abs(f17(1) - 111716437500.10001) < 1e-4);
  }

  SECTION("data desc id") {
    REQUIRE(pc::MeasurementSet::const_iterator::value_type(0, ms).data_desc_id().size() == 12852);
    REQUIRE(pc::MeasurementSet::const_iterator::value_type(17, ms).data_desc_id().size() == 12852);
  }

  SECTION("I") {
    using namespace purify;
    auto const I = pc::MeasurementSet::const_iterator::value_type(17, ms).I();
    REQUIRE(I.size() == 12852);
    CHECK(std::abs(I(0) - (t_complex(70.09417724609375,0) + t_complex(67.90460968017578125,0))) < 1e-4);
    CHECK(std::abs(I(10) - (t_complex(69.5322723388671875,0) + t_complex(68.62453460693359375,0))) < 1e-4);

    REQUIRE(pc::MeasurementSet::const_iterator::value_type(0, ms).I().size() == 12852);
  }

  SECTION("wI") {
    using namespace purify;
    auto const wI = pc::MeasurementSet::const_iterator::value_type(17, ms).wI();
    REQUIRE(wI.size() == 12852);
    CAPTURE(wI.head(5).transpose());
    CHECK(wI.isApprox(0.0363696478307247161865234375 * 0.5 * purify::Vector<t_real>::Ones(wI.size())));
  }

  SECTION("frequencies") {
    using namespace purify;
    auto const f = pc::MeasurementSet::const_iterator::value_type(17, ms).frequencies();
    REQUIRE(f.size() == 12852);
    CHECK(std::abs(f(0) - 1413080112.831275463104248046875) < 1e-0);
    CHECK(std::abs(f(1680) - 1413080112.831275463104248046875) < 1e-0);
    CHECK(std::abs(f(3360) - 1413080112.831275463104248046875) < 1e-0);
    CHECK(std::abs(f(5040) - 1413080112.831275463104248046875) < 1e-0);
  }
}

TEST_CASE("Measurement channel") {
  using namespace purify;
  //REQUIRE(not purify::casa::MeasurementSet(purify::notinstalled::ngc5921_2_ms())[0].is_valid());
  auto const channel = purify::casa::MeasurementSet(purify::notinstalled::ngc5921_2_ms())[17];
  REQUIRE(channel.is_valid());
  auto const I = channel.I();
  REQUIRE(I.size() == 12852);
  CHECK(std::abs(I(0) - (t_complex(70.09417724609375,0) + t_complex(67.90460968017578125,0))) < 1e-4);
  CHECK(std::abs(I(10) - (t_complex(69.5322723388671875,0) + t_complex(68.62453460693359375,0))) < 1e-4);
}

TEST_CASE("Channel iteration") {
  std::vector<int> const valids{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,
    35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
    53,  54,  55,  56,  57,  58,  59,  60,  61,  62, 63};
  auto const ms = purify::casa::MeasurementSet(purify::notinstalled::ngc5921_2_ms());
  auto i_channel = ms.begin();
  auto const i_end = ms.end();
  for(; i_channel < i_end; i_channel += 10) {
    CAPTURE(i_channel->channel());
    CHECK(i_channel->channel() < 63);
    bool const is_valid
      = std::find(valids.begin(), valids.end(), i_channel->channel()) != valids.end();
    CHECK(is_valid == i_channel->is_valid());
  }
}

TEST_CASE("Read Measurement") {
  if (boost::filesystem::exists(purify::notinstalled::ngc5921_2_vis())) {
    purify::utilities::vis_params const vis_file =
    purify::utilities::read_visibility(purify::notinstalled::ngc5921_2_vis());
    purify::utilities::vis_params const ms_file =
    purify::casa::read_measurementset(purify::notinstalled::ngc5921_2_ms());

    REQUIRE(vis_file.u.size() == ms_file.u.size());
    CAPTURE(vis_file.u.tail(5));
    CAPTURE(ms_file.u.tail(5));
    CAPTURE(ms_file.u(1000)/vis_file.u(1000));
    CAPTURE(ms_file.vis(1000)/vis_file.vis(1000));
    CHECK(vis_file.u.isApprox(ms_file.u, 1e-3));
    CHECK(vis_file.v.isApprox(ms_file.v, 1e-3));
    CHECK(vis_file.vis.isApprox(ms_file.vis, 1e-6));
    CHECK(vis_file.weights.real().isApprox(ms_file.weights.real(), 1e-6));
  }
}

TEST_CASE("Direction") {
  auto const ms = purify::casa::MeasurementSet(purify::notinstalled::ngc5921_2_ms());
  auto const direction = ms.direction();
  auto const right_ascension = ms.right_ascension();
  auto const declination = ms.declination();
  CHECK(std::abs(right_ascension - -2.260201381332657799561047795577906072139739990234375) < 1e-8);
  CHECK(std::abs(declination - 0.0884300154343793665123740765920956619083881378173828125) < 1e-8);
  CHECK(std::abs(direction[0] - -2.260201381332657799561047795577906072139739990234375) < 1e-8);
  CHECK(std::abs(direction[1] - 0.0884300154343793665123740765920956619083881378173828125) < 1e-8);
}

namespace purify{
TEST_CASE("Reading channels"){
  purify::logging::initialize();
  purify::utilities::vis_params const uv_data = purify::casa::read_measurementset(ngc5921_2_ms(), purify::casa::MeasurementSet::ChannelWrapper::polarization::I, std::vector<t_int>());
  std::vector<purify::utilities::vis_params> const uv_channels = purify::casa::read_measurementset_channels(ngc5921_2_ms(), purify::casa::MeasurementSet::ChannelWrapper::polarization::I, 1);
  t_int vis_i = 0;
  for (t_int channel = 0; channel < uv_channels.size(); channel++) {
    auto const channel_data = uv_channels[channel].vis;
    CHECK(not channel_data.isApprox(purify::Vector<t_complex>::Zero(channel_data.size())));
    for (t_int i = 0; i < channel_data.size(); i++) {
      //Testing if data in channels is the same as reading all the data at once.
      CHECK(uv_data.vis(vis_i) == channel_data(i));
      vis_i++;
    }
  }
  //Check that all data has been compared.
  CHECK(vis_i == uv_data.vis.size());
}
};
