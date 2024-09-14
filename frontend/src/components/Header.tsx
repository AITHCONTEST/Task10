
import { observer } from "mobx-react-lite";
import styles from "./styles/header.module.scss"
import MenuButt from "./menu/MenuButt";


const Header = observer(() => {

  
    return (
      <header className={styles.header}>

        <MenuButt/>

        <div className={styles.main}>
          Translate<span>2Win</span>
        </div>

        {/* <div className={styles.title}> Русско-Мансийский Переводчик </div> */}
      </header>
    );
  });
  
  export default Header;
  