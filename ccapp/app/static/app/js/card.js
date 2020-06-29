$(function(){
    var idNo = 1;
    // 追加ボタン押下時イベント
    $('button#add_rate_in_rightsandcps').on('click',function(){
      var process_btn = $('add_rate_in_rightsandcps').val();
      console.log(process_btn);
      $('div#temp_rate_in_rightsandcps')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "rate_in_rightsandcps" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayarea_rate_in_rightsandcps");
      // ID番号加算
      idNo++;
    });

    // 削除ボタン押下時イベント
    $('button[name=removeButton]').on('click',function(){
        $(this).parent('div').remove();
    });
});

$(function(){
    var idNo = 1;
    // 追加ボタン押下時イベント
    $('button#add_arrived').on('click',function(){
      var process_btn = $('button#add_arrived').val();
      console.log(process_btn);
      $('div#arrivedForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_canceled').on('click',function(){
      var process_btn = $('button#add_canceled').val();
      console.log(process_btn);
      $('div#canceledForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_delivered').on('click',function(){
      var process_btn = $('button#add_delivered').val();
      console.log(process_btn);
      $('div#deliveredForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });


    $('button#add_duedate').on('click',function(){
      var process_btn = $('button#add_duedate').val();
      console.log(process_btn);
      $('div#duedateForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_offered').on('click',function(){
      var process_btn = $('button#add_offered').val();
      console.log(process_btn);
      $('div#offeredForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_ordered').on('click',function(){
      var process_btn = $('button#add_ordered').val();
      console.log(process_btn);
      $('div#orderedForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_proprieter').on('click',function(){
      var process_btn = $('button#add_proprieter').val();
      console.log(process_btn);
      $('div#proprieterForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_remark').on('click',function(){
      var process_btn = $('button#add_remark').val();
      console.log(process_btn);
      $('div#remarkForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_requested').on('click',function(){
      var process_btn = $('button#add_requested').val();
      console.log(process_btn);
      $('div#requestedForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_shipped').on('click',function(){
      var process_btn = $('button#add_shipped').val();
      console.log(process_btn);
      $('div#shippedForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });

    $('button#add_submitted').on('click',function(){
      var process_btn = $('button#add_submitted').val();
      console.log(process_btn);
      $('div#submittedForm')
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "auctionhistory_input" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayArea");
      // ID番号加算
      idNo++;
    });


    // 削除ボタン押下時イベント
    $('button[name=removeButton]').on('click',function(){
        $(this).parent('div').remove();
    });
});
